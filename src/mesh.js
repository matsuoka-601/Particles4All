

import * as MS from './mesh_wgsl.js';
import { anisoWGSL } from './aniso_wgsl.js';

const WG = 256;

function dispatch2D(pass, count) {
  const groups = Math.ceil(count / WG);
  const gx = Math.min(groups, 32768);
  const gy = Math.ceil(groups / gx);
  pass.dispatchWorkgroups(gx, gy);
  return groups;
}

export class SurfaceMesh {
  constructor(device, format) {
    this.dev = device;
    this.format = format;
    this.maxTriangles = 3000000;
    this.normalSmoothPasses = 0;

    const mk = (src, label) => device.createShaderModule({
      code: MS.meshPrelude + src, label });
    const cp = (module, label) => device.createComputePipeline({
      label, layout: 'auto', compute: { module, entryPoint: 'main' } });
    this.pDensity = cp(mk(MS.meshDensityWGSL, 'meshDensity'), 'meshDensity');
    this.pSmooth = cp(mk(MS.meshSmoothWGSL, 'meshSmooth'), 'meshSmooth');
    this.pMarch = cp(mk(MS.meshMarchWGSL, 'meshMarch'), 'meshMarch');
    this.pIndirect = cp(mk(MS.meshIndirectWGSL, 'meshIndirect'), 'meshIndirect');

    this.pAniso = device.createComputePipeline({
      label: 'aniso', layout: 'auto',
      compute: { module: device.createShaderModule({ code: anisoWGSL, label: 'aniso' }),
                 entryPoint: 'main' } });
    this.anisoUni = device.createBuffer({ size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.aF = new Float32Array(20);
    this.aI = new Int32Array(this.aF.buffer);
    this.aU = new Uint32Array(this.aF.buffer);
    this.anisoCapacity = 0;
    this.anisoGen = 0;

    const dm = device.createShaderModule({ code: MS.meshDrawWGSL, label: 'meshDraw' });
    this.pipeDraw = device.createRenderPipeline({
      label: 'meshDraw', layout: 'auto',
      vertex: { module: dm, entryPoint: 'vs' },
      fragment: { module: dm, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.pipeSurf = device.createRenderPipeline({
      label: 'meshSurface', layout: 'auto',
      vertex: { module: dm, entryPoint: 'vs' },
      fragment: { module: dm, entryPoint: 'fsSurf',
                  targets: [{ format: 'r32float' }, { format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.uni = device.createBuffer({ size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uF = new Float32Array(28);
    this.uI = new Int32Array(this.uF.buffer);
    this.drawUni = device.createBuffer({ size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.drawF = new Float32Array(8);
    this.drawI = new Int32Array(this.drawF.buffer);
    this.counter = device.createBuffer({ size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.indirect = device.createBuffer({ size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });

    this.triRing = [0, 1, 2].map(() => device.createBuffer({ size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    this.triState = [0, 0, 0];
    this.frame = 0;
    this.lastTriangles = 0;

    this.anisoRatio = 4.0;
    this.anisoKs = 1.0;
    this.anisoStretch = 2.0;
    this.anisoLonely = 1.0;
    this.anisoMinNeighbours = 25;
    this.anisoLambda = 0.9;
    this.anisoRadiusScale = 2.0;

    this.anisoLimitToField = true;
    this.fieldCapacity = 0;
    this.vertCapacity = 0;

    this.gen = 0;
  }

  configure(boxMin, boxMax, resolution, maxTriangles = this.maxTriangles) {
    const size = [boxMax[0] - boxMin[0], boxMax[1] - boxMin[1], boxMax[2] - boxMin[2]];
    const longest = Math.max(...size);
    const voxel = longest / resolution;
    const voxelDim = size.map(s => Math.ceil(s / voxel) + 2);
    const vertDim = voxelDim.map(v => v + 1);
    const nVerts = vertDim[0] * vertDim[1] * vertDim[2];
    const fieldBytes = nVerts * 4;
    const vertBytes = maxTriangles * 3 * 8;
    if (fieldBytes > this.fieldCapacity) {
      this.field?.destroy(); this.field2?.destroy(); this.nfield?.destroy();

      this.fieldCapacity = 4 * Math.ceil((fieldBytes + fieldBytes / 4) / 4);
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      this.field = this.dev.createBuffer({ size: this.fieldCapacity, usage, label: 'field' });
      this.field2 = this.dev.createBuffer({ size: this.fieldCapacity, usage, label: 'field2' });
      this.nfield = this.dev.createBuffer({ size: this.fieldCapacity, usage, label: 'nfield' });
      this.gen++;
    }
    if (vertBytes !== this.vertCapacity) {
      this.verts?.destroy();
      this.vertCapacity = vertBytes;
      this.verts = this.dev.createBuffer({ size: vertBytes,
        usage: GPUBufferUsage.STORAGE, label: 'meshVerts' });
      this.gen++;
    }
    this.boxMin = boxMin;
    this.voxel = voxel;
    this.vertDim = vertDim;
    this.nVerts = nVerts;
    this.vertBudget = maxTriangles;
    this.origin = [boxMin[0] - voxel, boxMin[1] - voxel, boxMin[2] - voxel];
  }

  setParams(sim, iso, axis) {
    const F = this.uF, I = this.uI;
    I[0] = this.vertDim[0]; I[1] = this.vertDim[1]; I[2] = this.vertDim[2];
    F[3] = this.voxel;
    F[4] = this.origin[0]; F[5] = this.origin[1]; F[6] = this.origin[2];
    F[7] = iso;
    F[8] = 0; F[9] = 0; F[10] = 0;
    F[11] = 1 / sim.h;
    I[12] = axis[0]; I[13] = axis[1]; I[14] = axis[2];
    F[15] = sim.h * sim.h;
    I[16] = sim.gridDim[0]; I[17] = sim.gridDim[1]; I[18] = sim.gridDim[2];
    F[19] = 315 / (64 * Math.PI * Math.pow(sim.h, 9));
    F[20] = sim.scene.mass;
    F[21] = sim.params.restDensity;
    I[22] = this.vertBudget;
    I[23] = sim.nBodyParts > 0 ? 1 : 0;
    this.dev.queue.writeBuffer(this.uni, 0, this.uF);
  }

  bindGroups(sim) {
    const key = `${sim.gen}|${sim.parity}|${this.gen}`;
    if (this.bindCache && this.bindCache.key === key) return this.bindCache;
    const dev = this.dev;
    const s = sim.parity === 0 ? 'A' : 'B';
    const g = {
      density: dev.createBindGroup({ layout: this.pDensity.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: sim.buf['pos' + s] } },
        { binding: 2, resource: { buffer: sim.buf.density } },
        { binding: 3, resource: { buffer: sim.buf.cellStart } },
        { binding: 4, resource: { buffer: this.field } },
        { binding: 5, resource: { buffer: sim.buf['body' + s] } }] }),

      smoothAB: dev.createBindGroup({ layout: this.pSmooth.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.field } },
        { binding: 2, resource: { buffer: this.field2 } }] }),
      smoothBA: dev.createBindGroup({ layout: this.pSmooth.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.field2 } },
        { binding: 2, resource: { buffer: this.field } }] }),

      smoothFN: dev.createBindGroup({ layout: this.pSmooth.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.field } },
        { binding: 2, resource: { buffer: this.nfield } }] }),
      smoothNF2: dev.createBindGroup({ layout: this.pSmooth.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.nfield } },
        { binding: 2, resource: { buffer: this.field2 } }] }),
      smoothF2N: dev.createBindGroup({ layout: this.pSmooth.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.field2 } },
        { binding: 2, resource: { buffer: this.nfield } }] }),
      march: dev.createBindGroup({ layout: this.pMarch.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.field } },
        { binding: 2, resource: { buffer: this.verts } },
        { binding: 3, resource: { buffer: this.counter } },
        { binding: 4, resource: { buffer: this.field } }] }),

      marchN: dev.createBindGroup({ layout: this.pMarch.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.field } },
        { binding: 2, resource: { buffer: this.verts } },
        { binding: 3, resource: { buffer: this.counter } },
        { binding: 4, resource: { buffer: this.nfield } }] }),
      indirect: dev.createBindGroup({ layout: this.pIndirect.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.counter } },
        { binding: 2, resource: { buffer: this.indirect } }] }),
      key,
    };
    this.bindCache = g;
    return g;
  }

  buildAnisotropy(enc, sim) {
    const n = sim.n;
    if (n <= 0) return;
    const bytes = n * 2 * 4 * 4;
    if (bytes > this.anisoCapacity) {
      this.smoothPos?.destroy(); this.aniso?.destroy();
      this.anisoCapacity = bytes;

      const u = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
      this.smoothPos = this.dev.createBuffer({ size: n * 4 * 4, usage: u,
                                              label: 'smoothPos' });
      this.aniso = this.dev.createBuffer({ size: bytes, usage: u, label: 'aniso' });
      this.anisoGen++;
    }
    const support = sim.h;

    const radius = this.anisoRadiusScale * support;
    const cells = Math.ceil(this.anisoRadiusScale);

    const sigmaBulk = 0.15 * radius * radius;
    const F = this.aF, I = this.aI, U = this.aU;
    U[0] = n;
    F[1] = this.anisoLambda;
    F[2] = 1 / support;
    F[3] = support;
    F[4] = 0; F[5] = 0; F[6] = 0;
    F[7] = radius;
    I[8] = sim.gridDim[0]; I[9] = sim.gridDim[1]; I[10] = sim.gridDim[2];
    I[11] = cells;
    F[12] = this.anisoRatio;
    F[13] = this.anisoLonely;
    F[14] = this.anisoKs / sigmaBulk;
    F[15] = this.anisoStretch;
    F[16] = this.anisoLimitToField ? 1.25 * this.voxel / support : 0;
    I[17] = this.anisoMinNeighbours;
    this.dev.queue.writeBuffer(this.anisoUni, 0, this.aF);

    const key = `${sim.gen}|${sim.parity}|${this.anisoGen}`;
    if (!this.anisoBind || this.anisoBindKey !== key) {
      this.anisoBindKey = key;
      this.anisoBind = this.dev.createBindGroup({
        layout: this.pAniso.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.anisoUni } },
                  { binding: 1, resource: { buffer: sim.buf[sim.parity === 0 ? 'posA' : 'posB'] } },
                  { binding: 2, resource: { buffer: sim.buf.cellStart } },
                  { binding: 3, resource: { buffer: this.smoothPos } },
                  { binding: 4, resource: { buffer: this.aniso } }],
      });
    }
    const pass = enc.beginComputePass(this.timer?.mark('aniso'));
    pass.setPipeline(this.pAniso);
    pass.setBindGroup(0, this.anisoBind);
    dispatch2D(pass, n);
    pass.end();
  }

  generateField(enc, sim, iso) {
    const g = this.bindGroups(sim);
    this.setParams(sim, iso, [1, 0, 0]);
    const pass = enc.beginComputePass(this.timer?.mark('field'));
    pass.setPipeline(this.pDensity);
    pass.setBindGroup(0, g.density);
    dispatch2D(pass, this.nVerts);
    pass.end();
  }

  smoothField(enc, sim, iso, passes) {
    if (passes <= 0) return;
    const g = this.bindGroups(sim);
    let srcIsField = true;
    for (let p = 0; p < passes; p++) {
      for (let a = 0; a < 3; a++) {
        this.setParams(sim, iso, [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0]);
        const pass = enc.beginComputePass(this.timer?.mark('fieldblur'));
        pass.setPipeline(this.pSmooth);
        pass.setBindGroup(0, srcIsField ? g.smoothAB : g.smoothBA);
        dispatch2D(pass, this.nVerts);
        pass.end();
        srcIsField = !srcIsField;
      }
    }

    if (!srcIsField) {
      enc.copyBufferToBuffer(this.field2, 0, this.field, 0, this.nVerts * 4);
    }
  }

  buildNormalField(enc, sim, iso, passes) {
    this.normalFieldReady = false;
    if (passes <= 0) return;
    const g = this.bindGroups(sim);
    const steps = 3 * passes;

    let toNormal = (steps % 2) === 1;
    for (let s = 0; s < steps; s++) {
      const a = s % 3;
      this.setParams(sim, iso, [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0]);
      const pass = enc.beginComputePass(this.timer?.mark('normals'));
      pass.setPipeline(this.pSmooth);
      if (s === 0) pass.setBindGroup(0, toNormal ? g.smoothFN : g.smoothAB);
      else pass.setBindGroup(0, toNormal ? g.smoothF2N : g.smoothNF2);
      dispatch2D(pass, this.nVerts);
      pass.end();
      toNormal = !toNormal;
    }
    this.normalFieldReady = true;
  }

  generate(enc, sim, iso, smoothPasses) {
    const g = this.bindGroups(sim);
    this.generateField(enc, sim, iso);
    this.smoothField(enc, sim, iso, smoothPasses);
    this.buildNormalField(enc, sim, iso, this.normalSmoothPasses);
    this.setParams(sim, iso, [1, 0, 0]);
    enc.clearBuffer(this.counter);
    {
      const pass = enc.beginComputePass(this.timer?.mark('march'));
      pass.setPipeline(this.pMarch);

      pass.setBindGroup(0, this.normalFieldReady ? g.marchN : g.march);
      const nvox = this.vertDim.map(v => v - 1);
      dispatch2D(pass, nvox[0] * nvox[1] * nvox[2]);
      pass.setPipeline(this.pIndirect);
      pass.setBindGroup(0, g.indirect);
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    const slot = this.frame % 3;
    if (this.triState[slot] === 0) {
      enc.copyBufferToBuffer(this.counter, 0, this.triRing[slot], 0, 4);
      this.triState[slot] = 1;
    }
    this.frame++;
  }

  pollTriangles() {
    for (let i = 0; i < 3; i++) {
      if (this.triState[i] !== 1) continue;
      this.triState[i] = 2;
      const buf = this.triRing[i];
      buf.mapAsync(GPUMapMode.READ).then(() => {
        this.lastTriangles = new Uint32Array(buf.getMappedRange())[0];
        buf.unmap();
        this.triState[i] = 0;
      }, () => { this.triState[i] = 0; });
    }
  }

  resizeSurface(w, h) {
    if (this.surfW === w && this.surfH === h) return;
    this.surfZ?.destroy(); this.surfN?.destroy();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    this.surfZ = this.dev.createTexture({ size: [Math.max(1, w), Math.max(1, h)],
      format: 'r32float', usage, label: 'surfZ' });
    this.surfN = this.dev.createTexture({ size: [Math.max(1, w), Math.max(1, h)],
      format: 'rgba16float', usage, label: 'surfN' });
    this.surfW = w; this.surfH = h;
    this.surfViews = { z: this.surfZ.createView(), n: this.surfN.createView() };
  }

  surfBind(viewUni) {
    const key = `${this.gen}`;
    if (this.surfBindGroup && this.surfBindKey === key) return this.surfBindGroup;
    this.surfBindKey = key;
    this.surfBindGroup = this.dev.createBindGroup({
      layout: this.pipeSurf.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: viewUni } },
                { binding: 1, resource: { buffer: this.drawUni } },
                { binding: 2, resource: { buffer: this.verts } }],
    });
    return this.surfBindGroup;
  }

  drawSurface(enc, viewUni, depthView, w, h) {
    this.resizeSurface(w, h);
    this.writeDrawUni();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: this.surfViews.z, clearValue: { r: -1e4, g: 0, b: 0, a: 0 },
          loadOp: 'clear', storeOp: 'store' },
        { view: this.surfViews.n, clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: depthView, depthClearValue: 1.0,
        depthLoadOp: 'clear', depthStoreOp: 'store' },
      ...this.timer?.mark('surfgbuf'),
    });
    pass.setPipeline(this.pipeSurf);
    pass.setBindGroup(0, this.surfBind(viewUni));
    pass.drawIndirect(this.indirect, 0);
    pass.end();
    return true;
  }

  writeDrawUni() {
    this.drawF[0] = this.origin[0]; this.drawF[1] = this.origin[1];
    this.drawF[2] = this.origin[2]; this.drawF[3] = this.voxel;
    this.drawI[4] = this.vertDim[0]; this.drawI[5] = this.vertDim[1];
    this.drawI[6] = this.vertDim[2];
    this.dev.queue.writeBuffer(this.drawUni, 0, this.drawF);
  }

  drawBind(viewUni) {
    const key = `${this.gen}`;
    if (this.drawBindGroup && this.drawBindKey === key) return this.drawBindGroup;
    this.drawBindKey = key;
    this.drawBindGroup = this.dev.createBindGroup({
      layout: this.pipeDraw.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: viewUni } },
                { binding: 1, resource: { buffer: this.drawUni } },
                { binding: 2, resource: { buffer: this.verts } }],
    });
    return this.drawBindGroup;
  }

  draw(pass, viewUni) {
    this.writeDrawUni();
    pass.setPipeline(this.pipeDraw);
    pass.setBindGroup(0, this.drawBind(viewUni));
    pass.drawIndirect(this.indirect, 0);
  }
}
