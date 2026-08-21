

import * as S from './wgsl.js';

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0,
          0, f, 0, 0,
          0, 0, (far + near) * nf, -1,
          0, 0, 2 * far * near * nf, 0];
}
function lookAt(eye, target, up) {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return [x[0], y[0], z[0], 0,
          x[1], y[1], z[1], 0,
          x[2], y[2], z[2], 0,
          -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
const norm = a => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
function mul(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}

function inverse(m) {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  const d = det ? 1 / det : 0;
  return [(a11 * b11 - a12 * b10 + a13 * b09) * d,
          (a02 * b10 - a01 * b11 - a03 * b09) * d,
          (a31 * b05 - a32 * b04 + a33 * b03) * d,
          (a22 * b04 - a21 * b05 - a23 * b03) * d,
          (a12 * b08 - a10 * b11 - a13 * b07) * d,
          (a00 * b11 - a02 * b08 + a03 * b07) * d,
          (a32 * b02 - a30 * b05 - a33 * b01) * d,
          (a20 * b05 - a22 * b02 + a23 * b01) * d,
          (a10 * b10 - a11 * b08 + a13 * b06) * d,
          (a01 * b08 - a00 * b10 - a03 * b06) * d,
          (a30 * b04 - a31 * b02 + a33 * b00) * d,
          (a21 * b02 - a20 * b04 - a23 * b00) * d,
          (a11 * b07 - a10 * b09 - a12 * b06) * d,
          (a00 * b09 - a01 * b07 + a02 * b06) * d,
          (a31 * b01 - a30 * b03 - a32 * b00) * d,
          (a20 * b03 - a21 * b01 + a22 * b00) * d];
}

export const FOVY = 45 * Math.PI / 180;

export class Camera {
  constructor() { this.az = -0.5; this.el = 0.35; this.dist = 2.2;
                  this.target = [0.5, 0.4, 0.3]; }

  frame(box) {
    this.target = [box[0] / 2, box[1] / 2, box[2] / 2];
    this.dist = 1.8 * Math.max(box[0], box[1], box[2]);
  }
  eyeDir() {
    const ce = Math.cos(this.el), se = Math.sin(this.el);
    return [ce * Math.sin(this.az), se, ce * Math.cos(this.az)];
  }
  eye() {
    const d = this.eyeDir();
    return [this.target[0] + this.dist * d[0],
            this.target[1] + this.dist * d[1],
            this.target[2] + this.dist * d[2]];
  }
  orbit(dAz, dEl) {
    this.az += dAz;
    this.el = Math.max(-1.5, Math.min(1.5, this.el + dEl));
  }
  zoom(f) { this.dist = Math.max(0.3, Math.min(15, this.dist * f)); }
  pan(dx, dy) {
    const eyeDir = this.eyeDir();
    const right = norm(cross(eyeDir, [0, 1, 0]));
    const up = norm(cross(right, eyeDir));
    for (let i = 0; i < 3; i++)
      this.target[i] += (right[i] * dx + up[i] * dy) * this.dist;
  }
}

export function screenRay(cam, u, v, aspect, fovy = FOVY) {
  const eye = cam.eye();
  const forward = norm(sub(cam.target, eye));
  const right = norm(cross(forward, [0, 1, 0]));
  const up = norm(cross(right, forward));
  const tanHalfFov = Math.tan(fovy * 0.5);
  const sx = (2 * u - 1) * aspect * tanHalfFov;
  const sy = (1 - 2 * v) * tanHalfFov;
  const dir = norm([forward[0] + sx * right[0] + sy * up[0],
                    forward[1] + sx * right[1] + sy * up[1],
                    forward[2] + sx * right[2] + sy * up[2]]);
  return { origin: eye, dir };
}

export class Renderer {
  constructor(device, context, format) {
    this.dev = device;
    this.ctx = context;
    this.format = format;

    this.view = new Float32Array(64);
    this.uni = device.createBuffer({ size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const pmod = device.createShaderModule({ code: S.particleWGSL, label: 'particles' });
    this.pipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: pmod, entryPoint: 'vs' },
      fragment: { module: pmod, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    const bmod = device.createShaderModule({ code: S.boxWGSL, label: 'box' });
    this.boxPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: bmod, entryPoint: 'vs' },
      fragment: { module: bmod, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.boxBuf = device.createBuffer({ size: 24 * 2 * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.boxBind = device.createBindGroup({
      layout: this.boxPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uni } },
                { binding: 1, resource: { buffer: this.boxBuf } }],
    });
  }

  setBox(box) {
    const [x, y, z] = box;
    const c = [[0, 0, 0], [x, 0, 0], [x, 0, z], [0, 0, z],
               [0, y, 0], [x, y, 0], [x, y, z], [0, y, z]];
    const e = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
               [0, 4], [1, 5], [2, 6], [3, 7]];
    const v = new Float32Array(e.length * 2 * 4);
    e.forEach((pair, i) => pair.forEach((idx, k) => {
      v[(i * 2 + k) * 4 + 0] = c[idx][0];
      v[(i * 2 + k) * 4 + 1] = c[idx][1];
      v[(i * 2 + k) * 4 + 2] = c[idx][2];
    }));
    this.dev.queue.writeBuffer(this.boxBuf, 0, v);
    this.boxVerts = e.length * 2;
  }

  bindSim(sim) {
    this.simBind = [0, 1].map(par => {
      const pos = sim.buf[par === 0 ? 'posA' : 'posB'];
      const vel = sim.buf[par === 0 ? 'velA' : 'velB'];
      const body = sim.buf[par === 0 ? 'bodyA' : 'bodyB'];
      return this.dev.createBindGroup({
        layout: this.pipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.uni } },
                  { binding: 1, resource: { buffer: pos } },
                  { binding: 2, resource: { buffer: vel } },
                  { binding: 3, resource: { buffer: body } }],
      });
    });
  }

  project(pt, cam, aspect, fovy = FOVY) {
    const eye = cam.eye();
    const fwd = norm(sub(cam.target, eye));
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);
    const d = sub(pt, eye);
    const z = dot(d, fwd);
    if (z <= 1e-4) return null;
    const t = Math.tan(fovy / 2);
    const x = dot(d, right) / (z * t * aspect);
    const y = dot(d, up) / (z * t);
    return [(x + 1) * 0.5, (1 - y) * 0.5];
  }

  resize(w, h) {
    if (this.depth) this.depth.destroy();
    this.depth = this.dev.createTexture({
      size: [Math.max(1, w), Math.max(1, h)], format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.w = w; this.h = h;
  }

  draw(sim, cam, opts) {
    const eye = cam.eye();

    const proj = perspective(FOVY, this.w / Math.max(1, this.h), 0.05, 100);
    const viewM = lookAt(eye, cam.target, [0, 1, 0]);
    const vp = mul(proj, viewM);
    this.view.set(vp, 0);
    this.view.set(viewM, 16);
    this.view.set(proj, 32);
    this.view.set([eye[0], eye[1], eye[2], 0], 48);

    const fwd = norm(sub(cam.target, eye));
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = norm(cross(right, fwd));
    this.view.set([right[0], right[1], right[2], 0], 52);
    this.view.set([up[0], up[1], up[2], 0], 56);
    this.view[60] = opts.radius;
    this.view[61] = opts.speedMax;
    new Uint32Array(this.view.buffer, 248, 1)[0] = 0;
    this.dev.queue.writeBuffer(this.uni, 0, this.view);

    const enc = this.dev.createCommandEncoder();

    const T = this.timer;
    T?.begin();

    const mesh = opts.mesh;

    const ssfrOn = opts.display === 3 && mesh && opts.ssfr;
    const meshOn = opts.display !== 0 && !ssfrOn && mesh;
    const rayOn = opts.display === 2 && meshOn && opts.ray;
    const solids = opts.solids;
    const target = this.ctx.getCurrentTexture().createView();
    const depthView = this.depth.createView();

    if (meshOn) {
      const budget = (rayOn && opts.raySurface === 0) ? 200000 : mesh.maxTriangles;
      mesh.configure([0, 0, 0], sim.params.box, opts.meshRes, budget);
      if (rayOn && opts.raySurface === 0) {
        mesh.generateField(enc, sim, opts.meshIso);
        mesh.smoothField(enc, sim, opts.meshIso, opts.fieldSmooth);
      } else {
        mesh.generate(enc, sim, opts.meshIso, opts.fieldSmooth);
      }
    }

    if (solids) solids.pack(enc, sim);

    if (ssfrOn) {
      mesh.anisoLimitToField = false;
      mesh.configure([0, 0, 0], sim.params.box, opts.meshRes, 200000);
      mesh.buildAnisotropy(enc, sim);
      const invViewProj = inverse(vp);
      const invView = [right[0], right[1], right[2], 0,
                       up[0], up[1], up[2], 0,
                       -fwd[0], -fwd[1], -fwd[2], 0,
                       eye[0], eye[1], eye[2], 1];
      opts.ssfr.render(enc, target, sim, mesh, solids, viewM, proj, invViewProj,
                       invView, eye, this.w, this.h,
                       sim.params.spacing, sim.h);
      T?.resolve(enc);
      this.dev.queue.submit([enc.finish()]);
      T?.poll();
      return;
    }
    mesh.anisoLimitToField = true;

    if (rayOn) {
      const invViewProj = inverse(vp);

      const invView = [right[0], right[1], right[2], 0,
                       up[0], up[1], up[2], 0,
                       -fwd[0], -fwd[1], -fwd[2], 0,
                       eye[0], eye[1], eye[2], 1];
      let surf = false;
      if (opts.raySurface === 1) {
        mesh.drawSurface(enc, this.uni, depthView, this.w, this.h);
        surf = true;
      } else {
        mesh.resizeSurface(this.w, this.h);
      }
      opts.ray.draw(enc, target, mesh, sim, cam, viewM, proj, invViewProj, invView,
                    solids?.packed, solids?.count || 0, opts.meshIso, surf,
                    solids?.gen || 0);
      T?.resolve(enc);
      this.dev.queue.submit([enc.finish()]);
      T?.poll();
      mesh.pollTriangles();
      return;
    }

    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: target,
        clearValue: { r: 0.09, g: 0.10, b: 0.12, a: 1 },
        loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: {
        view: depthView, depthClearValue: 1.0,
        depthLoadOp: 'clear', depthStoreOp: 'store' },
      ...T?.mark('draw'),
    });
    if (meshOn) {
      mesh.draw(pass, this.uni);

      if (solids) solids.draw(pass, this.uni);
    } else {
      pass.setPipeline(this.pipe);
      pass.setBindGroup(0, this.simBind[sim.parity]);
      pass.draw(6, sim.n);
    }
    pass.setPipeline(this.boxPipe);
    pass.setBindGroup(0, this.boxBind);
    pass.draw(this.boxVerts);
    pass.end();
    T?.resolve(enc);
    this.dev.queue.submit([enc.finish()]);
    T?.poll();
    if (meshOn) mesh.pollTriangles();
  }
}
