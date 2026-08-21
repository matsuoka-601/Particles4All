

import * as SW from './ssfr_wgsl.js';
import * as CW from './ssfr_composite_wgsl.js';

const SPLAT_BYTES = 224;

const COMP_BYTES = 256;

export class FluidSSFR {
  constructor(device, format) {
    this.dev = device;
    this.format = format;

    this.ior = 1.333;
    this.absorption = 1.0;
    this.transmit = [0.35, 0.62, 0.78];
    this.roughness = 0.055;
    this.sunIntensity = 3.0;
    this.sunElevation = 38.0;
    this.sunAzimuth = 40.0;
    this.exposure = 1.0;
    this.groundReflection = 0.0;
    this.floorPlane = true;

    this.splatRadius = 1.0;

    this.renderScale = 1.0;

    this.filter = 2;
    this.filterSigma = 0.7;
    this.narrowDelta = 10.0;
    this.narrowMu = 1.0;
    this.bilateralRange = 2.0;
    this.filterIterations = 2;
    this.cleanupPass = true;
    this.cleanupRadius = 4;
    this.filterMaxRadius = 32;

    this.thicknessRadius = 0.62;
    this.thicknessScale = 3.0;
    this.thicknessFilterSize = 8;
    this.thicknessHalfRes = true;
    this.depthCullFraction = 0.0;
    this.depthCullNeighbours = 0;
    this.debug = 0;

    const splatMod = (src, label) => device.createShaderModule({
      code: SW.splatPrelude + src, label });
    const splatPipe = (module, targets, depth) => device.createRenderPipeline({
      label: 'ssfrSplat', layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets },

      primitive: { topology: 'triangle-strip' },
      depthStencil: depth,
    });
    this.pipeDepth = splatPipe(splatMod(SW.depthFS, 'ssfrDepth'),
      [{ format: 'r32float' }],
      { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' });
    this.pipeThick = splatPipe(splatMod(SW.thickFS, 'ssfrThick'),

      [{ format: 'r16float',
         blend: { color: { srcFactor: 'one', dstFactor: 'one' },
                  alpha: { srcFactor: 'one', dstFactor: 'one' } } }],
      undefined);

    const fullPipe = (code, label, fmt) => device.createRenderPipeline({
      label, layout: 'auto',
      vertex: { module: device.createShaderModule({ code, label }), entryPoint: 'vs' },
      fragment: { module: device.createShaderModule({ code, label: label + 'FS' }),
                  entryPoint: 'fs', targets: [{ format: fmt }] },
      primitive: { topology: 'triangle-list' },
    });
    this.pipeThickBlur = fullPipe(SW.thickFilterWGSL, 'ssfrThickBlur', 'r16float');
    this.pipeFilter = fullPipe(SW.filterWGSL, 'ssfrFilter', 'r32float');
    this.pipeComposite = fullPipe(CW.compositePrelude + CW.compositeFS,
                                  'ssfrComposite', format);
    this.pipeBodyT = fullPipe(CW.compositePrelude + CW.bodyDepthFS,
                              'ssfrSolidT', 'r32float');

    const uni = bytes => device.createBuffer({ size: bytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.splatUni = [uni(SPLAT_BYTES), uni(SPLAT_BYTES)];
    this.sF = new Float32Array(SPLAT_BYTES / 4);
    this.sU = new Uint32Array(this.sF.buffer);
    this.filtUni = [uni(48), uni(48), uni(48)];
    this.fF = new Float32Array(12);
    this.fI = new Int32Array(this.fF.buffer);
    this.blurUni = [uni(16), uni(16)];
    this.bI = new Int32Array(4);
    this.compUni = uni(COMP_BYTES);
    this.cF = new Float32Array(COMP_BYTES / 4);
    this.cI = new Int32Array(this.cF.buffer);
    this.thickSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }

  buildTargets(w, h) {
    const half = this.thicknessHalfRes && this.thicknessFilterSize >= 4;
    const tw = half ? (w + 1) >> 1 : w;
    const th = half ? (h + 1) >> 1 : h;
    if (this.w === w && this.h === h && this.tw === tw && this.th === th) return;
    for (const t of [this.eyeZ0, this.eyeZ1, this.raw, this.thick, this.thickTmp,
                     this.depthTex, this.bodyT]) t?.destroy();
    const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const mk = (fmt, ww, hh, extra = 0) => this.dev.createTexture({
      size: [Math.max(1, ww), Math.max(1, hh)], format: fmt, usage: RT | extra });
    this.eyeZ0 = mk('r32float', w, h, GPUTextureUsage.COPY_SRC);
    this.eyeZ1 = mk('r32float', w, h, GPUTextureUsage.COPY_SRC);
    this.raw = mk('r32float', w, h, GPUTextureUsage.COPY_DST);
    this.thick = mk('r16float', tw, th);
    this.thickTmp = mk('r16float', tw, th);
    this.bodyT = mk('r32float', w, h);
    this.depthTex = this.dev.createTexture({ size: [Math.max(1, w), Math.max(1, h)],
      format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.w = w; this.h = h; this.tw = tw; this.th = th;
    this.gen = (this.gen || 0) + 1;
    this.views = {
      eyeZ: [this.eyeZ0.createView(), this.eyeZ1.createView()],
      raw: this.raw.createView(),
      thick: this.thick.createView(),
      thickTmp: this.thickTmp.createView(),
      bodyT: this.bodyT.createView(),
      depth: this.depthTex.createView(),
    };
    this.bindCache = null;
  }

  writeSplat(slot, view, proj, invViewProj, vw, vh, scale, quadRadius,
             thickRadius, maxNeighbours, skipBodies) {
    const F = this.sF, U = this.sU;
    F.set(view, 0);
    F.set(proj, 16);
    F.set(invViewProj, 32);
    F[48] = vw; F[49] = vh;
    F[50] = scale;
    F[51] = quadRadius;
    F[52] = thickRadius;
    F[53] = maxNeighbours;
    U[54] = skipBodies ? 1 : 0;
    F[55] = 0;
    this.dev.queue.writeBuffer(this.splatUni[slot], 0, this.sF);
  }

  writeFilt(slot, dirX, dirY, clean, proj11, viewH, r) {
    const F = this.fF, I = this.fI;
    I[0] = dirX; I[1] = dirY;
    I[2] = this.filterMaxRadius;
    I[3] = this.filter;
    F[4] = this.filterSigma * r;
    F[5] = proj11;
    F[6] = viewH;
    F[7] = this.narrowDelta * r;
    F[8] = this.narrowMu * r;
    F[9] = Math.max(this.bilateralRange * r, 1e-4);
    I[10] = clean;
    I[11] = this.cleanupRadius;
    this.dev.queue.writeBuffer(this.filtUni[slot], 0, this.fF);
  }

  render(enc, target, sim, mesh, solids, view, proj, invViewProj, invView, eye,
         width, height, spacing, support) {
    const rs = Math.min(1, Math.max(0.25, this.renderScale));
    const mapW = Math.max(1, Math.round(width * rs));
    const mapH = Math.max(1, Math.round(height * rs));
    this.buildTargets(mapW, mapH);
    const dev = this.dev;

    const T = this.timer;
    const el = this.sunElevation * Math.PI / 180, az = this.sunAzimuth * Math.PI / 180;
    const sun = [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
    const sl = Math.hypot(...sun);
    const t = this.transmit.map(v => Math.min(1, Math.max(1e-3, v)));
    const absorb = t.map(v => this.absorption * -Math.log(v));

    const scale = this.splatRadius * spacing / support;
    const r = 0.5 * spacing;
    const thickRadius = this.thicknessRadius * spacing;
    const env = this.env;
    const skipBodies = sim.nBodyParts > 0;
    const bodyCount = solids?.count || 0;
    const bodies = solids?.packed || mesh.field;

    this.writeSplat(0, view, proj, invViewProj, mapW, mapH, scale, 0,
                    thickRadius, this.depthCullNeighbours, skipBodies);

    this.writeSplat(1, view, proj, invViewProj, this.tw, this.th, scale,
                    thickRadius, thickRadius, 0, skipBodies);
    {
      const F = this.cF, I = this.cI;
      F.set(invViewProj, 0);
      F.set(invView, 16);
      F.set([eye[0], eye[1], eye[2], 0], 32);
      F.set([0, 0, 0, proj[0]], 36);
      F.set([sim.params.box[0], sim.params.box[1], sim.params.box[2], proj[5]], 40);
      F.set([absorb[0], absorb[1], absorb[2], this.ior], 44);
      F.set([sun[0] / sl, sun[1] / sl, sun[2] / sl, this.sunIntensity], 48);
      F[52] = this.roughness;
      F[53] = this.exposure;
      F[54] = this.groundReflection;
      F[55] = this.thicknessScale;
      I[56] = bodyCount;
      I[57] = this.floorPlane ? 1 : 0;
      I[58] = this.debug;
      I[59] = env?.has ? 1 : 0;
      F[60] = env ? env.intensity : 1;
      F[61] = env ? env.yaw : 0;

      F[62] = mapW / Math.max(1, width);
      F[63] = mapH / Math.max(1, height);
      dev.queue.writeBuffer(this.compUni, 0, this.cF);
    }
    const blurSize = this.tw < this.w ? Math.max(1, this.thicknessFilterSize >> 1)
                                     : this.thicknessFilterSize;
    this.bI[2] = blurSize;
    for (let p = 0; p < 2; p++) {
      this.bI[0] = p === 0 ? 1 : 0;
      this.bI[1] = p === 0 ? 0 : 1;
      dev.queue.writeBuffer(this.blurUni[p], 0, this.bI);
    }

    this.writeFilt(0, 1, 0, 0, proj[5], mapH, r);
    this.writeFilt(1, 0, 1, 0, proj[5], mapH, r);
    this.writeFilt(2, 0, 0, 1, proj[5], mapH, r);

    const key = `${sim.gen}|${sim.parity}|${mesh.anisoGen}|${this.gen}` +
                `|${solids?.gen || 0}|${env?.gen || 0}`;
    if (!this.bindCache || this.bindCache.key !== key) {
      const splatEntries = slot => [
        { binding: 0, resource: { buffer: this.splatUni[slot] } },
        { binding: 1, resource: { buffer: mesh.smoothPos } },
        { binding: 2, resource: { buffer: mesh.aniso } },
        { binding: 3, resource: { buffer: sim.buf[sim.parity === 0 ? 'bodyA' : 'bodyB'] } }];
      this.bindCache = {
        key,
        depth: dev.createBindGroup({ layout: this.pipeDepth.getBindGroupLayout(0),
                                     entries: splatEntries(0) }),
        thick: dev.createBindGroup({ layout: this.pipeThick.getBindGroupLayout(0),
                                     entries: splatEntries(1) }),

        bodyT: dev.createBindGroup({ layout: this.pipeBodyT.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: this.compUni } },
                    { binding: 1, resource: { buffer: bodies } }] }),
        blur: [0, 1].map(p => dev.createBindGroup({
          layout: this.pipeThickBlur.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: this.blurUni[p] } },
                    { binding: 1, resource: p === 0 ? this.views.thick
                                                    : this.views.thickTmp }] })),

        filt: [0, 1, 2].map(slot => [0, 1].map(src => dev.createBindGroup({
          layout: this.pipeFilter.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: this.filtUni[slot] } },
                    { binding: 1, resource: this.views.eyeZ[src] }] }))),
        comp: [0, 1].map(src => dev.createBindGroup({
          layout: this.pipeComposite.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: this.compUni } },
                    { binding: 1, resource: { buffer: bodies } },
                    { binding: 2, resource: this.views.eyeZ[src] },
                    { binding: 3, resource: this.views.raw },
                    { binding: 4, resource: this.views.thick },
                    { binding: 5, resource: this.views.bodyT },
                    { binding: 6, resource: this.thickSampler },
                    { binding: 7, resource: env.view },
                    { binding: 8, resource: env.sampler }] })),
      };
    }
    const B = this.bindCache;

    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: this.views.bodyT,
          clearValue: { r: 1e30, g: 0, b: 0, a: 0 },
          loadOp: 'clear', storeOp: 'store' }],
        ...T?.mark('solidT'),
      });
      if (bodyCount > 0) {
        pass.setPipeline(this.pipeBodyT);
        pass.setBindGroup(0, B.bodyT);
        pass.draw(3);
      }
      pass.end();
    }

    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: this.views.eyeZ[0],
          clearValue: { r: -1e4, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
        depthStencilAttachment: { view: this.views.depth, depthClearValue: 1.0,
          depthLoadOp: 'clear', depthStoreOp: 'store' },
        ...T?.mark('depth'),
      });
      pass.setPipeline(this.pipeDepth);
      pass.setBindGroup(0, B.depth);
      pass.draw(4, sim.n);
      pass.end();
    }

    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: this.views.thick,
          clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
        ...T?.mark('thick'),
      });
      pass.setPipeline(this.pipeThick);
      pass.setBindGroup(0, B.thick);
      pass.draw(4, sim.n);
      pass.end();
    }

    if (this.thicknessFilterSize > 0) {
      for (let p = 0; p < 2; p++) {
        const pass = enc.beginRenderPass({
          colorAttachments: [{ view: p === 0 ? this.views.thickTmp : this.views.thick,
            loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
          ...T?.mark('thickblur'),
        });
        pass.setPipeline(this.pipeThickBlur);
        pass.setBindGroup(0, B.blur[p]);
        pass.draw(3);
        pass.end();
      }
    }

    T?.gap('copy');
    enc.copyTextureToTexture({ texture: this.eyeZ0 }, { texture: this.raw },
                             [this.w, this.h, 1]);

    let src = 0;
    const iterations = Math.max(this.filterIterations, 0);
    const filterPass = (slot) => {
      const dst = 1 - src;
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: this.views.eyeZ[dst], loadOp: 'clear',
          clearValue: { r: -1e4, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
        ...T?.mark('filter'),
      });
      pass.setPipeline(this.pipeFilter);
      pass.setBindGroup(0, B.filt[slot][src]);
      pass.draw(3);
      pass.end();
      src = dst;
    };
    for (let it = 0; it < iterations * 2; it++) filterPass(it % 2 === 0 ? 0 : 1);
    if (this.cleanupPass && iterations > 0) filterPass(2);

    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: target, clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear', storeOp: 'store' }],
        ...T?.mark('composite'),
      });
      pass.setPipeline(this.pipeComposite);
      pass.setBindGroup(0, B.comp[src]);
      pass.draw(3);
      pass.end();
    }
  }
}
