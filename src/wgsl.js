

export const prelude = `
struct Params {
  boxMin      : vec3f,
  dt          : f32,
  boxMax      : vec3f,
  h           : f32,
  gridDim     : vec3i,
  h2          : f32,
  clampMin    : vec3f,
  poly6       : f32,
  clampMax    : vec3f,
  spikyGrad   : f32,
  gravity     : f32,
  mass        : f32,
  rho0        : f32,
  invRho0     : f32,
  cfmEps      : f32,
  sCorrK      : f32,
  sCorrWq     : f32,
  xsphC       : f32,
  n           : u32,
  nCells      : u32,
  nBoundary   : u32,
  nBodyParts  : u32,
  nBodies     : u32,
  bodyScale   : f32,
  covScale    : f32,
  omega       : f32,
  sorAverage  : u32,
  invDt       : f32,
  volume      : f32,
  tension     : f32,
  cohes       : f32,
  cohesTerm   : f32,
  pad0        : f32,
  pad1        : f32,
  pad2        : f32,
}
@group(0) @binding(0) var<uniform> P : Params;

fn cellOf(p: vec3f) -> vec3i {
  let c = vec3i(floor((p - P.boxMin) / P.h));
  return clamp(c, vec3i(0), P.gridDim - vec3i(1));
}
fn cellIndex(c: vec3i) -> u32 {
  return u32((c.z * P.gridDim.y + c.y) * P.gridDim.x + c.x);
}
fn poly6(r2: f32) -> f32 {
  let t = P.h2 - r2;
  return select(0.0, P.poly6 * t * t * t, r2 < P.h2);
}
`;

export const predictWGSL = `
@group(0) @binding(1) var<storage, read_write> pos  : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> vel  : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> pred : array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }

  let v = vel[i].xyz + P.dt * vec3f(0.0, -P.gravity, 0.0);
  vel[i] = vec4f(v, 0.0);
  pred[i] = vec4f(pos[i].xyz + P.dt * v, 1.0);
}
`;

export const velFromPosWGSL = `
@group(0) @binding(1) var<storage, read>       pos  : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> vel  : array<vec4f>;
@group(0) @binding(3) var<storage, read>       pred : array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  vel[i] = vec4f((pred[i].xyz - pos[i].xyz) * P.invDt, 0.0);
}
`;

export const xsphWGSL = `
@group(0) @binding(1) var<storage, read>       pred      : array<vec4f>;
@group(0) @binding(2) var<storage, read>       vel       : array<vec4f>;
@group(0) @binding(3) var<storage, read>       density   : array<f32>;
@group(0) @binding(4) var<storage, read_write> corr      : array<vec4f>;
@group(0) @binding(5) var<storage, read>       cellStart : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pred[i].xyz;
  let vi = vel[i].xyz;
  let c = cellOf(pi);
  let x0 = max(c.x - 1, 0);
  let x1 = min(c.x + 1, P.gridDim.x - 1);
  var acc = vec3f(0.0);
  for (var dz = -1; dz <= 1; dz++) {
    let z = c.z + dz;
    if (z < 0 || z >= P.gridDim.z) { continue; }
    for (var dy = -1; dy <= 1; dy++) {
      let y = c.y + dy;
      if (y < 0 || y >= P.gridDim.y) { continue; }
      let rowBase = u32((z * P.gridDim.y + y) * P.gridDim.x);
      let b = cellStart[rowBase + u32(x0)];
      let e = cellStart[rowBase + u32(x1) + 1u];
      for (var j = b; j < e; j++) {
        if (j == i) { continue; }
        let rij = pi - pred[j].xyz;
        let r2 = dot(rij, rij);
        if (r2 >= P.h2) { continue; }
        let f = P.mass * poly6(r2) / max(density[j], 0.5 * P.rho0);
        acc += f * (vel[j].xyz - vi);
      }
    }
  }
  corr[i] = vec4f(P.xsphC * acc, 0.0);
}
`;

export const resizeWGSL = `
struct Box {
  oldMin  : vec3f,
  pad0    : f32,
  oldSize : vec3f,
  pad1    : f32,
  newMin  : vec3f,
  pad2    : f32,
  newSize : vec3f,
  pad3    : f32,
  clampMin: vec3f,
  pad4    : f32,
  clampMax: vec3f,
  n       : u32,
}
@group(0) @binding(0) var<uniform> B : Box;
@group(0) @binding(1) var<storage, read_write> pos : array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= B.n) { return; }
  var p = pos[i].xyz;
  for (var a = 0; a < 3; a++) {
    if (B.newSize[a] < B.oldSize[a]) {
      let t = clamp((p[a] - B.oldMin[a]) / B.oldSize[a], 0.0, 1.0);
      p[a] = B.newMin[a] + t * B.newSize[a];
    }
    p[a] = clamp(p[a], B.clampMin[a], B.clampMax[a]);
  }
  pos[i] = vec4f(p, pos[i].w);
}
`;

export const bodyClearWGSL = `
@group(0) @binding(1) var<storage, read_write> acc : array<i32>;
@group(0) @binding(2) var<storage, read_write> cov : array<i32>;
@group(0) @binding(3) var<storage, read_write> idxCount : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < arrayLength(&acc)) { acc[i] = 0; }
  if (i < arrayLength(&cov)) { cov[i] = 0; }

  if (i == 0u) { idxCount[0] = 0u; }
}
`;

export const bodyRefLoadWGSL = `

@group(0) @binding(1) var<storage, read>       refCentre : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> centre    : array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&refCentre)) { return; }
  centre[i] = refCentre[i];
}
`;

export const bodyDragWGSL = `
struct Drag {
  goal   : vec3f,
  rate   : f32,
  held   : u32,
  blend  : f32,
  limit  : f32,
  nBody  : u32,
}
@group(0) @binding(0) var<uniform> D : Drag;
@group(0) @binding(1) var<storage, read_write> vel    : array<vec4f>;
@group(0) @binding(2) var<storage, read>       body   : array<vec4u>;
@group(0) @binding(3) var<storage, read>       idx    : array<u32>;
@group(0) @binding(4) var<storage, read>       centre : array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= D.nBody) { return; }
  let i = idx[t];
  if (body[i].x != D.held) { return; }
  var vWant = (D.goal - centre[D.held - 1u].xyz) * D.rate;
  let sp = length(vWant);
  if (sp > D.limit) { vWant *= D.limit / sp; }
  let v = vel[i].xyz;
  vel[i] = vec4f(v + (vWant - v) * D.blend, vel[i].w);
}
`;

export const impulseWGSL = `
struct Ray {
  origin     : vec3f,
  radius     : f32,
  dir        : vec3f,
  speedLimit : f32,
  impulse    : vec3f,
  n          : u32,
}
@group(0) @binding(0) var<uniform> R : Ray;
@group(0) @binding(1) var<storage, read>       pos : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> vel : array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= R.n) { return; }
  let toP = pos[i].xyz - R.origin;
  let along = dot(toP, R.dir);
  if (along <= 0.0) { return; }
  let radial = toP - along * R.dir;
  let d2 = dot(radial, radial);
  if (d2 >= R.radius * R.radius) { return; }
  let x = 1.0 - sqrt(d2) / R.radius;
  let v = vel[i].xyz;
  let oldSpeed = length(v);
  var nv = v + (x * x) * R.impulse;
  let ns = length(nv);
  let allowed = max(oldSpeed, R.speedLimit);
  if (ns > allowed) { nv *= allowed / ns; }
  vel[i] = vec4f(nv, 0.0);
}
`;

export const normalsWGSL = `
@group(0) @binding(1) var<storage, read>       pred      : array<vec4f>;
@group(0) @binding(2) var<storage, read>       density   : array<f32>;
@group(0) @binding(3) var<storage, read_write> normalBuf : array<vec4f>;
@group(0) @binding(4) var<storage, read>       cellStart : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pred[i].xyz;
  let c = cellOf(pi);
  let x0 = max(c.x - 1, 0);
  let x1 = min(c.x + 1, P.gridDim.x - 1);
  var acc = vec3f(0.0);
  for (var dz = -1; dz <= 1; dz++) {
    let z = c.z + dz;
    if (z < 0 || z >= P.gridDim.z) { continue; }
    for (var dy = -1; dy <= 1; dy++) {
      let y = c.y + dy;
      if (y < 0 || y >= P.gridDim.y) { continue; }
      let rowBase = u32((z * P.gridDim.y + y) * P.gridDim.x);
      let b = cellStart[rowBase + u32(x0)];
      let e = cellStart[rowBase + u32(x1) + 1u];
      for (var j = b; j < e; j++) {
        if (j == i) { continue; }
        let rij = pi - pred[j].xyz;
        let r2 = dot(rij, rij);
        if (r2 >= P.h2 || r2 < 1.0e-12) { continue; }
        let r = sqrt(r2);
        let hr = P.h - r;
        acc += (P.mass / max(density[j], 0.5 * P.rho0)) *
               (P.spikyGrad * hr * hr / r) * rij;
      }
    }
  }
  normalBuf[i] = vec4f(P.h * acc, 0.0);
}
`;

export const tensionWGSL = `
@group(0) @binding(1) var<storage, read>       pred      : array<vec4f>;
@group(0) @binding(2) var<storage, read>       density   : array<f32>;
@group(0) @binding(3) var<storage, read>       normalBuf : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> corr      : array<vec4f>;
@group(0) @binding(5) var<storage, read>       cellStart : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pred[i].xyz;
  let ni = normalBuf[i].xyz;
  let rhoi = density[i];
  let c = cellOf(pi);
  let x0 = max(c.x - 1, 0);
  let x1 = min(c.x + 1, P.gridDim.x - 1);
  var acc = vec3f(0.0);
  for (var dz = -1; dz <= 1; dz++) {
    let z = c.z + dz;
    if (z < 0 || z >= P.gridDim.z) { continue; }
    for (var dy = -1; dy <= 1; dy++) {
      let y = c.y + dy;
      if (y < 0 || y >= P.gridDim.y) { continue; }
      let rowBase = u32((z * P.gridDim.y + y) * P.gridDim.x);
      let b = cellStart[rowBase + u32(x0)];
      let e = cellStart[rowBase + u32(x1) + 1u];
      for (var j = b; j < e; j++) {
        if (j == i) { continue; }
        let rij = pi - pred[j].xyz;
        let r2 = dot(rij, rij);
        if (r2 >= P.h2 || r2 < 1.0e-12) { continue; }
        let r = sqrt(r2);
        let hrr = P.h - r;
        let k = hrr * hrr * hrr * r * r * r;
        var cw = P.cohes * (2.0 * k - P.cohesTerm);
        if (2.0 * r > P.h) { cw = P.cohes * k; }
        let fCoh = (-P.tension * P.mass * cw / r) * rij;
        let fCurv = -P.tension * (ni - normalBuf[j].xyz);
        acc += (2.0 * P.rho0 / (rhoi + density[j])) * (fCoh + fCurv);
      }
    }
  }
  corr[i] = vec4f(corr[i].xyz + P.dt * acc, 0.0);
}
`;

export const finalizeWGSL = `
@group(0) @binding(1) var<storage, read_write> pos  : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> vel  : array<vec4f>;
@group(0) @binding(3) var<storage, read>       pred : array<vec4f>;
@group(0) @binding(4) var<storage, read>       corr : array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  vel[i] = vec4f(vel[i].xyz + corr[i].xyz, 0.0);
  pos[i] = pred[i];
}
`;

export const countWGSL = `
@group(0) @binding(1) var<storage, read>       pred      : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cellCount : array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  atomicAdd(&cellCount[cellIndex(cellOf(pred[i].xyz))], 1u);
}
`;

export const scanBlockWGSL = `
@group(0) @binding(1) var<storage, read>       cellCount : array<u32>;
@group(0) @binding(2) var<storage, read_write> cellStart : array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSum  : array<u32>;

var<workgroup> tmp : array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let i = gid.x;
  let t = lid.x;
  tmp[t] = select(0u, cellCount[i], i < P.nCells);
  workgroupBarrier();

  for (var off = 1u; off < 256u; off = off << 1u) {
    var v = tmp[t];
    if (t >= off) { v = v + tmp[t - off]; }
    workgroupBarrier();
    tmp[t] = v;
    workgroupBarrier();
  }
  if (i < P.nCells) {
    cellStart[i] = select(tmp[t - 1u], 0u, t == 0u);
  }
  if (t == 255u) { blockSum[wid.x] = tmp[255]; }
}
`;

export const scanBlocksWGSL = `
@group(0) @binding(1) var<storage, read_write> blockSum : array<u32>;

@compute @workgroup_size(1)
fn main() {
  let nBlocks = (P.nCells + 255u) / 256u;
  var run = 0u;
  for (var b = 0u; b < nBlocks; b++) {
    let v = blockSum[b];
    blockSum[b] = run;
    run = run + v;
  }

  blockSum[nBlocks] = run;
}
`;

export const scanAddWGSL = `
@group(0) @binding(1) var<storage, read_write> cellStart : array<u32>;
@group(0) @binding(2) var<storage, read>       blockSum  : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let i = gid.x;
  if (i > P.nCells) { return; }
  if (i == P.nCells) { cellStart[i] = blockSum[(P.nCells + 255u) / 256u]; return; }
  cellStart[i] = cellStart[i] + blockSum[wid.x];
}
`;

export const scatterWGSL = `
@group(0) @binding(1) var<storage, read>       pos       : array<vec4f>;
@group(0) @binding(2) var<storage, read>       vel       : array<vec4f>;
@group(0) @binding(3) var<storage, read>       pred      : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> pos2      : array<vec4f>;
@group(0) @binding(5) var<storage, read_write> vel2      : array<vec4f>;
@group(0) @binding(6) var<storage, read_write> pred2     : array<vec4f>;
@group(0) @binding(7) var<storage, read>       cellStart : array<u32>;
@group(0) @binding(8) var<storage, read_write> cursor    : array<atomic<u32>>;
@group(0) @binding(9) var<storage, read>       body      : array<vec4u>;
@group(0) @binding(10) var<storage, read>      rest      : array<vec4f>;
@group(0) @binding(11) var<storage, read_write> body2    : array<vec4u>;
@group(0) @binding(12) var<storage, read_write> rest2    : array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let cell = cellIndex(cellOf(pred[i].xyz));
  let slot = cellStart[cell] + atomicAdd(&cursor[cell], 1u);
  pos2[slot]  = pos[i];
  vel2[slot]  = vel[i];
  pred2[slot] = pred[i];
  body2[slot] = body[i];
  rest2[slot] = rest[i];
}
`;

export const lambdaWGSL = `
@group(0) @binding(1) var<storage, read>       pred       : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> lambda     : array<f32>;
@group(0) @binding(3) var<storage, read_write> density    : array<f32>;
@group(0) @binding(4) var<storage, read>       cellStart  : array<u32>;
@group(0) @binding(5) var<storage, read>       bpos       : array<vec4f>;
@group(0) @binding(6) var<storage, read>       bpsi       : array<f32>;
@group(0) @binding(7) var<storage, read>       bcellStart : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pred[i].xyz;
  let c = cellOf(pi);
  let x0 = max(c.x - 1, 0);
  let x1 = min(c.x + 1, P.gridDim.x - 1);
  var rho = P.mass * poly6(0.0);
  var gradSum = vec3f(0.0);
  var sumGrad2 = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    let z = c.z + dz;
    if (z < 0 || z >= P.gridDim.z) { continue; }
    for (var dy = -1; dy <= 1; dy++) {
      let y = c.y + dy;
      if (y < 0 || y >= P.gridDim.y) { continue; }
      let rowBase = u32((z * P.gridDim.y + y) * P.gridDim.x);
      let b = cellStart[rowBase + u32(x0)];
      let e = cellStart[rowBase + u32(x1) + 1u];
      for (var j = b; j < e; j++) {
        if (j == i) { continue; }
        let rij = pi - pred[j].xyz;
        let r2 = dot(rij, rij);
        if (r2 >= P.h2) { continue; }
        rho += P.mass * poly6(r2);
        if (r2 > 1.0e-12) {
          let r = sqrt(r2);
          let hr = P.h - r;
          let g = (P.volume * P.spikyGrad * hr * hr / r) * rij;
          gradSum += g;
          sumGrad2 += dot(g, g);
        }
      }
      if (P.nBoundary > 0u) {
        let bb = bcellStart[rowBase + u32(x0)];
        let be = bcellStart[rowBase + u32(x1) + 1u];
        for (var k = bb; k < be; k++) {
          let rij = pi - bpos[k].xyz;
          let r2 = dot(rij, rij);
          if (r2 >= P.h2 || r2 < 1.0e-12) { continue; }
          rho += bpsi[k] * poly6(r2);
          let r = sqrt(r2);
          let hr = P.h - r;
          gradSum += (bpsi[k] * P.invRho0 * P.spikyGrad * hr * hr / r) * rij;
        }
      }
    }
  }
  density[i] = rho;
  let C = rho * P.invRho0 - 1.0;
  if (C > 0.0) {
    lambda[i] = -C / (dot(gradSum, gradSum) + sumGrad2 + P.cfmEps + 1.0e-12);
  } else {
    lambda[i] = 0.0;
  }
}
`;

export const deltaWGSL = `
@group(0) @binding(1) var<storage, read>       pred       : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> predOut    : array<vec4f>;
@group(0) @binding(3) var<storage, read>       lambda     : array<f32>;
@group(0) @binding(4) var<storage, read>       cellStart  : array<u32>;
@group(0) @binding(5) var<storage, read>       bpos       : array<vec4f>;
@group(0) @binding(6) var<storage, read>       bpsi       : array<f32>;
@group(0) @binding(7) var<storage, read>       bcellStart : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pred[i].xyz;
  let li = lambda[i];
  let c = cellOf(pi);
  let x0 = max(c.x - 1, 0);
  let x1 = min(c.x + 1, P.gridDim.x - 1);

  var d = vec3f(0.0);
  var dB = vec3f(0.0);
  var nc = 1.0;
  for (var dz = -1; dz <= 1; dz++) {
    let z = c.z + dz;
    if (z < 0 || z >= P.gridDim.z) { continue; }
    for (var dy = -1; dy <= 1; dy++) {
      let y = c.y + dy;
      if (y < 0 || y >= P.gridDim.y) { continue; }
      let rowBase = u32((z * P.gridDim.y + y) * P.gridDim.x);
      let b = cellStart[rowBase + u32(x0)];
      let e = cellStart[rowBase + u32(x1) + 1u];
      for (var j = b; j < e; j++) {
        if (j == i) { continue; }
        let rij = pi - pred[j].xyz;
        let r2 = dot(rij, rij);
        if (r2 >= P.h2 || r2 < 1.0e-12) { continue; }
        let r = sqrt(r2);
        let hr = P.h - r;
        let s = P.spikyGrad * hr * hr / r;

        let t = P.h2 - r2;
        let ratio = P.poly6 * t * t * t * P.sCorrWq;
        let r2r = ratio * ratio;
        let sc = -P.sCorrK * r2r * r2r;
        d += (li + lambda[j] + sc) * s * rij;
        nc += 1.0;
      }
      if (P.nBoundary > 0u) {
        let bb = bcellStart[rowBase + u32(x0)];
        let be = bcellStart[rowBase + u32(x1) + 1u];
        for (var k = bb; k < be; k++) {
          let rij = pi - bpos[k].xyz;
          let r2 = dot(rij, rij);
          if (r2 >= P.h2 || r2 < 1.0e-12) { continue; }
          let r = sqrt(r2);
          let hr = P.h - r;
          dB += (li * bpsi[k]) * (P.spikyGrad * hr * hr / r) * rij;
        }
      }
    }
  }
  d = d * P.volume + dB * P.invRho0;

  var scale = P.omega;
  if (P.sorAverage == 1u) { scale = P.omega / max(nc, 1.0); }
  predOut[i] = vec4f(clamp(pi + d * scale, P.clampMin, P.clampMax), 1.0);
}
`;

export const bodyCompactWGSL = `
@group(0) @binding(1) var<storage, read>       body  : array<vec4u>;
@group(0) @binding(2) var<storage, read_write> idx   : array<u32>;
@group(0) @binding(3) var<storage, read_write> count : array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  if (body[i].x == 0u) { return; }
  idx[atomicAdd(&count[0], 1u)] = i;
}
`;

export const bodyCentreWGSL = `
@group(0) @binding(1) var<storage, read>       pred   : array<vec4f>;
@group(0) @binding(2) var<storage, read>       body   : array<vec4u>;
@group(0) @binding(3) var<storage, read_write> acc    : array<atomic<i32>>;
@group(0) @binding(4) var<storage, read>       idx    : array<u32>;
@group(0) @binding(5) var<storage, read>       centre : array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= P.nBodyParts) { return; }
  let i = idx[t];
  let b = body[i].x;
  if (b == 0u) { return; }
  let base = (b - 1u) * 4u;
  let d = (pred[i].xyz - centre[b - 1u].xyz) * P.bodyScale;
  atomicAdd(&acc[base + 0u], i32(round(d.x)));
  atomicAdd(&acc[base + 1u], i32(round(d.y)));
  atomicAdd(&acc[base + 2u], i32(round(d.z)));
  atomicAdd(&acc[base + 3u], 1);
}
`;

export const bodySeedWGSL = `
@group(0) @binding(1) var<storage, read>       acc    : array<i32>;
@group(0) @binding(2) var<storage, read_write> centre : array<vec4f>;
@group(0) @binding(3) var<storage, read>       refCentre : array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let b = gid.x;
  if (b >= P.nBodies) { return; }
  let a = b * 4u;
  let count = f32(acc[a + 3u]);
  if (count <= 0.0) { return; }
  let r = refCentre[b].xyz;
  let d = vec3f(f32(acc[a + 0u]), f32(acc[a + 1u]), f32(acc[a + 2u]))
          / (P.bodyScale * count);
  centre[b] = vec4f(r + d, 0.0);
}
`;

export const bodyCovWGSL = `
@group(0) @binding(1) var<storage, read>       pred   : array<vec4f>;
@group(0) @binding(2) var<storage, read>       body   : array<vec4u>;
@group(0) @binding(3) var<storage, read_write> cov    : array<atomic<i32>>;
@group(0) @binding(4) var<storage, read>       centre : array<vec4f>;
@group(0) @binding(5) var<storage, read>       rest   : array<vec4f>;
@group(0) @binding(6) var<storage, read>       idx    : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= P.nBodyParts) { return; }
  let i = idx[t];
  let b = body[i].x;
  if (b == 0u) { return; }
  let p = pred[i].xyz - centre[b - 1u].xyz;
  let q = rest[i].xyz;
  let base = (b - 1u) * 9u;
  atomicAdd(&cov[base + 0u], i32(round(p.x * q.x * P.covScale)));
  atomicAdd(&cov[base + 1u], i32(round(p.y * q.x * P.covScale)));
  atomicAdd(&cov[base + 2u], i32(round(p.z * q.x * P.covScale)));
  atomicAdd(&cov[base + 3u], i32(round(p.x * q.y * P.covScale)));
  atomicAdd(&cov[base + 4u], i32(round(p.y * q.y * P.covScale)));
  atomicAdd(&cov[base + 5u], i32(round(p.z * q.y * P.covScale)));
  atomicAdd(&cov[base + 6u], i32(round(p.x * q.z * P.covScale)));
  atomicAdd(&cov[base + 7u], i32(round(p.y * q.z * P.covScale)));
  atomicAdd(&cov[base + 8u], i32(round(p.z * q.z * P.covScale)));
}
`;

export const bodyResolveWGSL = `
@group(0) @binding(1) var<storage, read>       acc    : array<i32>;
@group(0) @binding(2) var<storage, read>       cov    : array<i32>;
@group(0) @binding(3) var<storage, read_write> centre : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> rot    : array<vec4f>;
@group(0) @binding(5) var<storage, read>       info   : array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let b = gid.x;
  if (b >= P.nBodies) { return; }
  let a = b * 4u;
  let count = f32(acc[a + 3u]);
  if (count <= 0.0) { return; }

  let prev = centre[b].xyz;
  var delta = vec3f(f32(acc[a + 0u]), f32(acc[a + 1u]), f32(acc[a + 2u]))
              / (P.bodyScale * count);

  let invRatio = info[b].x;
  if (invRatio > 0.0) { delta *= invRatio; }
  centre[b] = vec4f(prev + delta, 0.0);

  let k = b * 9u;
  var A = mat3x3f(f32(cov[k + 0u]), f32(cov[k + 1u]), f32(cov[k + 2u]),
                  f32(cov[k + 3u]), f32(cov[k + 4u]), f32(cov[k + 5u]),
                  f32(cov[k + 6u]), f32(cov[k + 7u]), f32(cov[k + 8u]));
  A = A * (1.0 / P.covScale);
  var det = determinant(A);
  if (abs(det) < 1.0e-12) { A = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); det = 1.0; }
  let s = pow(abs(det), 1.0 / 3.0);
  if (s > 1.0e-6) { A = A * (1.0 / s); }

  var R = A;
  for (var it = 0; it < 8; it++) {
    let inv = transpose(mat3x3fInverse(R));
    let next = (R + inv) * 0.5;
    let d0 = next[0] - R[0];
    let d1 = next[1] - R[1];
    let d2 = next[2] - R[2];
    R = next;
    if (dot(d0, d0) + dot(d1, d1) + dot(d2, d2) < 1.0e-12) { break; }
  }

  rot[b * 3u + 0u] = vec4f(R[0][0], R[1][0], R[2][0], 0.0);
  rot[b * 3u + 1u] = vec4f(R[0][1], R[1][1], R[2][1], 0.0);
  rot[b * 3u + 2u] = vec4f(R[0][2], R[1][2], R[2][2], 0.0);
}

fn mat3x3fInverse(m: mat3x3f) -> mat3x3f {
  let a = m[0]; let b = m[1]; let c = m[2];
  let r0 = cross(b, c);
  let r1 = cross(c, a);
  let r2 = cross(a, b);
  let d = dot(a, r0);
  let invd = select(0.0, 1.0 / d, abs(d) > 1.0e-20);

  return mat3x3f(vec3f(r0.x, r1.x, r2.x),
                 vec3f(r0.y, r1.y, r2.y),
                 vec3f(r0.z, r1.z, r2.z)) * invd;
}
`;

export const bodyProjectWGSL = `
@group(0) @binding(1) var<storage, read_write> pred   : array<vec4f>;
@group(0) @binding(2) var<storage, read>       body   : array<vec4u>;
@group(0) @binding(3) var<storage, read>       centre : array<vec4f>;
@group(0) @binding(4) var<storage, read>       rot    : array<vec4f>;
@group(0) @binding(5) var<storage, read>       rest   : array<vec4f>;
@group(0) @binding(6) var<storage, read>       idx    : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= P.nBodyParts) { return; }
  let i = idx[t];
  let b = body[i].x;
  if (b == 0u) { return; }
  let r = (b - 1u) * 3u;
  let q = rest[i].xyz;
  let p = vec3f(dot(rot[r + 0u].xyz, q), dot(rot[r + 1u].xyz, q), dot(rot[r + 2u].xyz, q));
  pred[i] = vec4f(centre[b - 1u].xyz + p, 1.0);
}
`;

export const statsWGSL = `
@group(0) @binding(1) var<storage, read>       density : array<f32>;
@group(0) @binding(2) var<storage, read>       vel     : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> out     : array<atomic<u32>>;

const SCALE : f32 = 1024.0;

var<workgroup> sRho : array<f32, 256>;
var<workgroup> sMaxRho : array<f32, 256>;
var<workgroup> sMaxV : array<f32, 256>;
var<workgroup> sKe : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u) {
  let i = gid.x;
  let t = lid.x;

  var rho = 0.0;
  var mrho = 0.0;
  var mv = 0.0;
  var ke = 0.0;
  if (i < P.n) {
    rho = density[i] * P.invRho0;
    mrho = rho;
    let v2 = dot(vel[i].xyz, vel[i].xyz);
    mv = sqrt(v2);
    ke = 0.5 * P.mass * v2;
  }
  sRho[t] = rho; sMaxRho[t] = mrho; sMaxV[t] = mv; sKe[t] = ke;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off = off >> 1u) {
    if (t < off) {
      sRho[t] = sRho[t] + sRho[t + off];
      sMaxRho[t] = max(sMaxRho[t], sMaxRho[t + off]);
      sMaxV[t] = max(sMaxV[t], sMaxV[t + off]);
      sKe[t] = sKe[t] + sKe[t + off];
    }
    workgroupBarrier();
  }
  if (t == 0u) {
    atomicAdd(&out[0], u32(sRho[0] * SCALE));
    atomicMax(&out[1], u32(sMaxRho[0] * SCALE));
    atomicMax(&out[2], u32(sMaxV[0] * SCALE));
    atomicAdd(&out[3], u32(sKe[0] * SCALE));
  }
}
`;

export const particleWGSL = `
struct View {
  viewProj : mat4x4f,
  view     : mat4x4f,
  proj     : mat4x4f,
  eye      : vec4f,
  right    : vec4f,
  up       : vec4f,
  radius   : f32,
  speedMax : f32,
  bodyOnly : u32,
  pad      : f32,
}
@group(0) @binding(0) var<uniform> V : View;
@group(0) @binding(1) var<storage, read> pos  : array<vec4f>;
@group(0) @binding(2) var<storage, read> vel  : array<vec4f>;
@group(0) @binding(3) var<storage, read> body : array<vec4u>;

struct VsOut {
  @builtin(position) clip : vec4f,
  @location(0) local : vec2f,
  @location(1) colour : vec3f,
  @location(2) viewCentre : vec3f,
  @location(3) radius : f32,
  @location(4) @interpolate(flat) isBody : u32,
}
struct FsOut {
  @location(0) colour : vec4f,
  @builtin(frag_depth) depth : f32,
}

const kBody = vec3f(1.0, 0.55, 0.10);

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
  var quad = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                             vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let q = quad[vi];
  let isBody = body[ii].x != 0u;
  var r = V.radius;
  if (isBody) { r = r * 1.35; }
  let c = pos[ii].xyz;
  let world = c + (V.right.xyz * q.x + V.up.xyz * q.y) * r;
  var o : VsOut;
  o.clip = V.viewProj * vec4f(world, 1.0);
  o.local = q;
  o.viewCentre = (V.view * vec4f(c, 1.0)).xyz;
  o.radius = r;
  let speed = length(vel[ii].xyz);
  let t = clamp(speed / max(V.speedMax, 1.0e-3), 0.0, 1.0);
  let slow = vec3f(0.10, 0.35, 0.85);
  let fast = vec3f(0.75, 0.92, 1.00);
  o.colour = select(mix(slow, fast, t), kBody, isBody);
  o.isBody = select(0u, 1u, isBody);
  if (V.bodyOnly == 1u && !isBody) { o.clip = vec4f(2.0, 2.0, 2.0, 1.0); }
  return o;
}

@fragment
fn fs(in: VsOut) -> FsOut {
  var uv = in.local;
  uv.y = -uv.y;
  let r2 = dot(uv, uv);
  if (r2 > 1.0) { discard; }
  let n = vec3f(uv, sqrt(1.0 - r2));

  let viewPos = in.viewCentre + n * in.radius;
  let clip = V.proj * vec4f(viewPos, 1.0);
  var o : FsOut;
  o.depth = clamp(clip.z / clip.w, 0.0, 1.0);

  let lightDir = normalize(vec3f(0.4, 0.8, 0.6));
  let diff = max(dot(n, lightDir), 0.0);
  let viewDir = normalize(-viewPos);
  let halfV = normalize(lightDir + viewDir);

  if (in.isBody == 1u) {
    let sb = pow(max(dot(n, halfV), 0.0), 64.0);
    o.colour = vec4f(in.colour * (0.30 + 0.70 * diff) + vec3f(0.5) * sb, 1.0);
    return o;
  }
  let spec = pow(max(dot(n, halfV), 0.0), 48.0);
  o.colour = vec4f(in.colour * (0.25 + 0.75 * diff) + vec3f(0.6) * spec, 1.0);
  return o;
}
`;

export const boxWGSL = `
struct View {
  viewProj : mat4x4f,
  view     : mat4x4f,
  proj     : mat4x4f,
  eye      : vec4f,
  right    : vec4f,
  up       : vec4f,
  radius   : f32,
  speedMax : f32,
  bodyOnly : u32,
  pad      : f32,
}
@group(0) @binding(0) var<uniform> V : View;
@group(0) @binding(1) var<storage, read> verts : array<vec4f>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return V.viewProj * vec4f(verts[vi].xyz, 1.0);
}
@fragment
fn fs() -> @location(0) vec4f { return vec4f(0.55, 0.55, 0.60, 1.0); }
`;
