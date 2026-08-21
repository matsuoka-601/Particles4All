

export const PRESETS = {
  small:   { ini: 'presets/small.ini' },
  medium:  { ini: 'presets/medium.ini' },
  large:   { ini: 'presets/large.ini' },
  extreme: { particles: 1000000, box: [3.0, 3.0, 3.0], substeps: 4, iterations: 4,
             bodies: ['sphere:0.5', 'torus:0.8', 'box:1.4'] },
};

export function defaultParams(preset = 'small') {
  const p = PRESETS[preset] || {};

  return Object.assign({
    preset,
    spacing: 0.02,
    restDensity: 1000,
    gravity: 9.81,
    substeps: 4,
    iterations: 4,
    cfmEpsilonRel: 0.01,
    sCorrK: 0.1,
    sCorrDq: 0.3,
    xsphC: 0.05,
    omega: 1.0,
    sorAverage: false,
    surfaceTensionK: 0,
    targetParticleCount: 60000,
    box: [1.5, 1.5, 1.0],
    bodies: ['sphere:0.5', 'torus:0.8', 'box:1.4'],
    bodySize: 0,
  }, {
    ...(p.substeps   !== undefined && { substeps: p.substeps }),
    ...(p.iterations !== undefined && { iterations: p.iterations }),
    ...(p.particles  !== undefined && { targetParticleCount: p.particles }),
    ...(p.box        !== undefined && { box: p.box.slice() }),
    ...(p.bodies     !== undefined && { bodies: p.bodies.slice() }),
  }, p.params || {});
}

const poly6Coef = h => 315 / (64 * Math.PI * Math.pow(h, 9));

function fluidBlock(p, d) {
  const [bx, by, bz] = p.box;
  const margin = d;
  const nx = Math.max(1, Math.floor(bx * 0.35 / d));
  const nz = Math.max(1, Math.floor((bz - 2 * margin) / d));
  const maxLayers = Math.max(1, Math.floor((by - 2 * margin) / d));
  const layers = Math.min(Math.ceil(p.targetParticleCount / (nx * nz)), maxLayers);
  const out = [];

  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff - 0.5) * 0.002 * d;
  };
  for (let iy = 0; iy < layers && out.length / 3 < p.targetParticleCount; iy++)
    for (let ix = 0; ix < nx && out.length / 3 < p.targetParticleCount; ix++)
      for (let iz = 0; iz < nz && out.length / 3 < p.targetParticleCount; iz++)
        out.push(margin + (ix + 0.5) * d + rnd(),
                 margin + (iy + 0.5) * d + rnd(),
                 margin + (iz + 0.5) * d + rnd());
  return out;
}

export function blockCapacity(p, d) {
  const [bx, by, bz] = p.box;
  const margin = d;
  return Math.max(1, Math.floor(bx * 0.35 / d)) *
         Math.max(1, Math.floor((bz - 2 * margin) / d)) *
         Math.max(1, Math.floor((by - 2 * margin) / d));
}

export function boundaryFor(p, d, h) { return boundaryParticles(p, d, h); }

function boundaryParticles(p, d, h) {
  const size = p.box;
  const nb = size.map(s => Math.max(2, Math.ceil(s / d) + 1));
  const step = size.map((s, k) => s / (nb[k] - 1));
  const pts = [];
  for (let ix = 0; ix < nb[0]; ix++)
    for (let iy = 0; iy < nb[1]; iy++)
      for (let iz = 0; iz < nb[2]; iz++) {
        const shell = ix === 0 || ix === nb[0] - 1 || iy === 0 || iy === nb[1] - 1 ||
                      iz === 0 || iz === nb[2] - 1;
        if (shell) pts.push(ix * step[0], iy * step[1], iz * step[2]);
      }

  const n = pts.length / 3;
  const dim = size.map(s => Math.max(1, Math.ceil(s / h)));
  const cellOf = (x, y, z) => [
    Math.min(dim[0] - 1, Math.max(0, Math.floor(x / h))),
    Math.min(dim[1] - 1, Math.max(0, Math.floor(y / h))),
    Math.min(dim[2] - 1, Math.max(0, Math.floor(z / h))),
  ];
  const key = (a, b, c) => (c * dim[1] + b) * dim[0] + a;
  const buckets = new Map();
  for (let i = 0; i < n; i++) {
    const [a, b, c] = cellOf(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
    const k = key(a, b, c);
    let arr = buckets.get(k);
    if (!arr) { arr = []; buckets.set(k, arr); }
    arr.push(i);
  }
  const coef = poly6Coef(h);
  const h2 = h * h;
  const psi = new Float32Array(Math.max(1, n));
  for (let i = 0; i < n; i++) {
    const [a, b, c] = cellOf(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
    let sum = 0;
    for (let dc = -1; dc <= 1; dc++) {
      if (c + dc < 0 || c + dc >= dim[2]) continue;
      for (let db = -1; db <= 1; db++) {
        if (b + db < 0 || b + db >= dim[1]) continue;
        for (let da = -1; da <= 1; da++) {
          if (a + da < 0 || a + da >= dim[0]) continue;
          const arr = buckets.get(key(a + da, b + db, c + dc));
          if (!arr) continue;
          for (const j of arr) {
            const dx = pts[i * 3] - pts[j * 3];
            const dy = pts[i * 3 + 1] - pts[j * 3 + 1];
            const dz = pts[i * 3 + 2] - pts[j * 3 + 2];
            const r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= h2) continue;
            const t = h2 - r2;
            sum += t * t * t;
          }
        }
      }
    }
    sum *= coef;
    psi[i] = sum > 0 ? p.restDensity / sum : 0;
  }
  return { pts, psi, count: n };
}

function sampleBody(shape, size, d) {
  const ext = shape === 'sphere' ? [size, size, size]
            : shape === 'torus'  ? [size + 0.4 * size, 0.4 * size, size + 0.4 * size]
                                 : [size * 0.9, size * 0.7, size * 0.9];
  const steps = ext.map(e => Math.max(1, Math.ceil(e / d)));
  const local = [];
  for (let ix = -steps[0]; ix <= steps[0]; ix++)
    for (let iy = -steps[1]; iy <= steps[1]; iy++)
      for (let iz = -steps[2]; iz <= steps[2]; iz++) {
        const x = ix * d, y = iy * d, z = iz * d;
        let inside = false;
        if (shape === 'sphere') inside = x * x + y * y + z * z <= size * size;
        else if (shape === 'torus') {
          const planar = Math.sqrt(x * x + z * z) - size;
          inside = planar * planar + y * y <= (0.4 * size) * (0.4 * size);
        } else inside = Math.abs(x) <= size * 0.9 && Math.abs(y) <= size * 0.7 &&
                        Math.abs(z) <= size * 0.9;
        if (inside) local.push(x, y, z);
      }
  return local;
}

export function buildScene(p) {
  const d = p.spacing;
  const h = 2 * d;
  const [bx, by, bz] = p.box;

  const fluid = fluidBlock(p, d);
  const nFluid = fluid.length / 3;

  const specs = p.bodies.map(s => {
    const bits = s.split(':');
    return { shape: bits[0], density: Number(bits[1] ?? 0.5) || 0.5,
             startY: Number(bits[2] ?? 0.75) || 0.75 };
  });
  const sz = p.bodySize > 0 ? p.bodySize : 0.12 * Math.min(bx, by, bz);
  const step = Math.min(2.5 * sz, bx * 0.25);
  const span = step * (specs.length - 1);

  const bodyPts = [];
  const bodyRest = [];
  const bodyPhase = [];
  const bodies = [];
  specs.forEach((spec, r) => {
    const local = sampleBody(spec.shape, sz, d);
    const count = local.length / 3;
    if (!count) return;
    const centre = [0.5 * bx - 0.5 * span + step * r, by * spec.startY, bz * 0.5];
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < count; i++) { cx += local[i * 3]; cy += local[i * 3 + 1]; cz += local[i * 3 + 2]; }
    cx /= count; cy /= count; cz /= count;
    const id = bodies.length + 1;
    for (let i = 0; i < count; i++) {
      const lx = local[i * 3], ly = local[i * 3 + 1], lz = local[i * 3 + 2];
      bodyPts.push(centre[0] + lx, centre[1] + ly, centre[2] + lz);
      bodyRest.push(lx - cx, ly - cy, lz - cz);
      bodyPhase.push(id);
    }
    bodies.push({ id, shape: spec.shape, size: sz, density: spec.density,
                  centre, count });
  });

  const nBody = bodyPts.length / 3;
  const n = nFluid + nBody;

  const pos = new Float32Array(n * 4);
  const vel = new Float32Array(n * 4);
  const rest = new Float32Array(n * 4);
  const phase = new Uint32Array(n * 4);
  for (let i = 0; i < nFluid; i++) {
    pos[i * 4 + 0] = fluid[i * 3 + 0];
    pos[i * 4 + 1] = fluid[i * 3 + 1];
    pos[i * 4 + 2] = fluid[i * 3 + 2];
  }
  for (let i = 0; i < nBody; i++) {
    const o = (nFluid + i) * 4;
    pos[o + 0] = bodyPts[i * 3 + 0];
    pos[o + 1] = bodyPts[i * 3 + 1];
    pos[o + 2] = bodyPts[i * 3 + 2];
    rest[o + 0] = bodyRest[i * 3 + 0];
    rest[o + 1] = bodyRest[i * 3 + 1];
    rest[o + 2] = bodyRest[i * 3 + 2];
    phase[o + 0] = bodyPhase[i];
  }

  const boundary = boundaryParticles(p, d, h);
  {
    let lo = Infinity, hi = 0;
    for (let i = 0; i < boundary.count; i++) {
      lo = Math.min(lo, boundary.psi[i]);
      hi = Math.max(hi, boundary.psi[i]);
    }
    console.info(`boundary ${boundary.count} samples, psi ${lo.toFixed(5)}..${hi.toFixed(5)}`);
  }

  const coef = poly6Coef(h);
  const h2 = h * h;
  const spiky = -45 / (Math.PI * Math.pow(h, 6));
  const dim = [Math.max(1, Math.floor(bx / h)), Math.max(1, Math.floor(by / h)),
               Math.max(1, Math.floor(bz / h))];
  const cellOf = i => [
    Math.min(dim[0] - 1, Math.max(0, Math.floor(pos[i * 4 + 0] / h))),
    Math.min(dim[1] - 1, Math.max(0, Math.floor(pos[i * 4 + 1] / h))),
    Math.min(dim[2] - 1, Math.max(0, Math.floor(pos[i * 4 + 2] / h))),
  ];
  const key = (a, b, c) => (c * dim[1] + b) * dim[0] + a;

  const buckets = new Map();
  for (let i = 0; i < nFluid; i++) {
    const [a, b, c] = cellOf(i);
    const k = key(a, b, c);
    let arr = buckets.get(k);
    if (!arr) { arr = []; buckets.set(k, arr); }
    arr.push(i);
  }
  const forEachNeighbour = (i, f) => {
    const [a, b, c] = cellOf(i);
    for (let dc = -1; dc <= 1; dc++) {
      if (c + dc < 0 || c + dc >= dim[2]) continue;
      for (let db = -1; db <= 1; db++) {
        if (b + db < 0 || b + db >= dim[1]) continue;
        for (let da = -1; da <= 1; da++) {
          if (a + da < 0 || a + da >= dim[0]) continue;
          const arr = buckets.get(key(a + da, b + db, c + dc));
          if (arr) for (const j of arr) if (j !== i) f(j);
        }
      }
    }
  };
  const densityAt = (i, m) => {
    let rho = m * coef * h2 * h2 * h2;
    forEachNeighbour(i, j => {
      const r2 = (pos[i * 4] - pos[j * 4]) ** 2 + (pos[i * 4 + 1] - pos[j * 4 + 1]) ** 2 +
                 (pos[i * 4 + 2] - pos[j * 4 + 2]) ** 2;
      if (r2 >= h2) return;
      const t = h2 - r2;
      rho += m * coef * t * t * t;
    });
    return rho;
  };
  let mass = p.restDensity * d * d * d;
  {
    let maxRho = 0;
    for (let i = 0; i < nFluid; i++) maxRho = Math.max(maxRho, densityAt(i, mass));
    if (maxRho > 0.5 * p.restDensity) {
      mass *= p.restDensity / maxRho;
    } else if (nFluid > 0) {
      console.info(`scene: no interior fluid particle to calibrate against ` +
                   `(densest ${maxRho.toFixed(1)} of ${p.restDensity}); ` +
                   `keeping m = rho0 d^3`);
    }
  }
  let denomRest = 0;
  {
    const volume = mass / p.restDensity;

    for (let i = 0; i < nFluid; i++) {
      if (densityAt(i, mass) < 0.99 * p.restDensity) continue;
      let gx = 0, gy = 0, gz = 0, sumGrad2 = 0;
      forEachNeighbour(i, j => {
        const rx = pos[i * 4] - pos[j * 4];
        const ry = pos[i * 4 + 1] - pos[j * 4 + 1];
        const rz = pos[i * 4 + 2] - pos[j * 4 + 2];
        const r2 = rx * rx + ry * ry + rz * rz;
        if (r2 < 1e-12 || r2 >= h2) return;
        const r = Math.sqrt(r2);
        const hr = h - r;
        const s = volume * spiky * hr * hr / r;
        gx += s * rx; gy += s * ry; gz += s * rz;
        sumGrad2 += s * s * r2;
      });
      denomRest = Math.max(denomRest, gx * gx + gy * gy + gz * gz + sumGrad2);
      if (i > 2000 && denomRest > 0) break;
    }
  }

  return { n, nFluid, nBody, pos, vel, rest, phase, boundary, bodies, mass, h, d,
           denomRest };
}
