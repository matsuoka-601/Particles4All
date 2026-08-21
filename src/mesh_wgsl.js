

export const meshPrelude = `
struct MeshParams {
  vertDim    : vec3i,
  voxel      : f32,
  gridOrigin : vec3f,
  iso        : f32,
  boxMin     : vec3f,
  invSupport : f32,
  axis       : vec3i,
  support2   : f32,
  simGridDim : vec3i,
  poly6      : f32,
  mass       : f32,
  rho0       : f32,
  maxTris    : u32,
  skipBodies : u32,
}
@group(0) @binding(0) var<uniform> M : MeshParams;
`;

export const meshDensityWGSL = `
@group(0) @binding(1) var<storage, read>       pos       : array<vec4f>;
@group(0) @binding(2) var<storage, read>       rho       : array<f32>;
@group(0) @binding(3) var<storage, read>       cellStart : array<u32>;
@group(0) @binding(4) var<storage, read_write> field     : array<f32>;
@group(0) @binding(5) var<storage, read>       body      : array<vec4u>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  let total = u32(M.vertDim.x * M.vertDim.y * M.vertDim.z);
  if (idx >= total) { return; }
  var v: vec3i;
  v.x = i32(idx) % M.vertDim.x;
  v.y = (i32(idx) / M.vertDim.x) % M.vertDim.y;
  v.z = i32(idx) / (M.vertDim.x * M.vertDim.y);
  let p = M.gridOrigin + M.voxel * vec3f(v);

  let c = clamp(vec3i(floor((p - M.boxMin) * M.invSupport)),
                vec3i(0), M.simGridDim - vec3i(1));
  let x0 = max(c.x - 1, 0);
  let x1 = min(c.x + 1, M.simGridDim.x - 1);
  var acc = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    let z = c.z + dz;
    if (z < 0 || z >= M.simGridDim.z) { continue; }
    for (var dy = -1; dy <= 1; dy++) {
      let y = c.y + dy;
      if (y < 0 || y >= M.simGridDim.y) { continue; }
      let rowBase = u32((z * M.simGridDim.y + y) * M.simGridDim.x);
      let b = cellStart[rowBase + u32(x0)];
      let e = cellStart[rowBase + u32(x1) + 1u];
      for (var j = b; j < e; j++) {
        if (M.skipBodies != 0u && body[j].x != 0u) { continue; }
        let vol = M.mass / max(rho[j], 0.5 * M.rho0);
        let rij = p - pos[j].xyz;
        let r2 = dot(rij, rij);
        if (r2 >= M.support2) { continue; }
        let t = M.support2 - r2;
        acc += vol * M.poly6 * t * t * t;
      }
    }
  }
  field[idx] = acc;
}
`;

export const meshSmoothWGSL = `
@group(0) @binding(1) var<storage, read>       src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;

fn S(cc: vec3i) -> f32 {
  let c = clamp(cc, vec3i(0), M.vertDim - vec3i(1));
  return src[u32((c.z * M.vertDim.y + c.y) * M.vertDim.x + c.x)];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  let total = u32(M.vertDim.x * M.vertDim.y * M.vertDim.z);
  if (idx >= total) { return; }
  var v: vec3i;
  v.x = i32(idx) % M.vertDim.x;
  v.y = (i32(idx) / M.vertDim.x) % M.vertDim.y;
  v.z = i32(idx) / (M.vertDim.x * M.vertDim.y);
  dst[idx] = 0.25 * S(v - M.axis) + 0.5 * S(v) + 0.25 * S(v + M.axis);
}
`;

export const meshMarchWGSL = `
@group(0) @binding(1) var<storage, read>        field   : array<f32>;
@group(0) @binding(2) var<storage, read_write>  verts   : array<vec2u>;
@group(0) @binding(3) var<storage, read_write>  counter : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read>        nfield  : array<f32>;

fn D(vv: vec3i) -> f32 {
  let v = clamp(vv, vec3i(0), M.vertDim - vec3i(1));
  return field[u32((v.z * M.vertDim.y + v.y) * M.vertDim.x + v.x)];
}
fn ND(vv: vec3i) -> f32 {
  let v = clamp(vv, vec3i(0), M.vertDim - vec3i(1));
  return nfield[u32((v.z * M.vertDim.y + v.y) * M.vertDim.x + v.x)];
}
fn gradAt(v: vec3i) -> vec3f {
  return vec3f(ND(v + vec3i(1,0,0)) - ND(v - vec3i(1,0,0)),
               ND(v + vec3i(0,1,0)) - ND(v - vec3i(0,1,0)),
               ND(v + vec3i(0,0,1)) - ND(v - vec3i(0,0,1)));
}

fn packVertex(pGrid: vec3f, nIn: vec3f) -> vec2u {
  let pn = clamp(pGrid / vec3f(M.vertDim - vec3i(1)), vec3f(0.0), vec3f(1.0));
  let q = vec3u(pn * 65535.0 + 0.5);
  var n = nIn / max(abs(nIn.x) + abs(nIn.y) + abs(nIn.z), 1.0e-12);
  var e = n.xy;
  if (n.z < 0.0) {
    e = (1.0 - abs(vec2f(n.y, n.x))) *
        vec2f(select(-1.0, 1.0, n.x >= 0.0), select(-1.0, 1.0, n.y >= 0.0));
  }
  let eo = vec2u(clamp(e * 0.5 + 0.5, vec2f(0.0), vec2f(1.0)) * 255.0 + 0.5);
  return vec2u(q.x | (q.y << 16u), q.z | ((eo.x | (eo.y << 8u)) << 16u));
}

fn edgeVertex(da: f32, pa: vec3f, ga: vec3f, db: f32, pb: vec3f, gb: vec3f) -> vec2u {
  let t = (M.iso - da) / (db - da);
  let p = mix(pa, pb, vec3f(t));
  let g = mix(ga, gb, vec3f(t));

  return packVertex(p, -g);
}

fn emitTri(a: vec2u, b: vec2u, c: vec2u) {
  let base = atomicAdd(&counter[0], 1u);
  if (base >= M.maxTris) { return; }
  verts[base * 3u + 0u] = a;
  verts[base * 3u + 1u] = b;
  verts[base * 3u + 2u] = c;
}

fn emitLone(dl: f32, pl: vec3f, gl: vec3f,
            d1: f32, p1: vec3f, g1: vec3f,
            d2: f32, p2: vec3f, g2: vec3f,
            d3: f32, p3: vec3f, g3: vec3f) {
  emitTri(edgeVertex(dl, pl, gl, d1, p1, g1),
          edgeVertex(dl, pl, gl, d2, p2, g2),
          edgeVertex(dl, pl, gl, d3, p3, g3));
}

fn emitQuad(da: f32, pa: vec3f, ga: vec3f,
            db: f32, pb: vec3f, gb: vec3f,
            dc: f32, pc: vec3f, gc: vec3f,
            dd: f32, pd: vec3f, gd: vec3f) {
  let q0 = edgeVertex(da, pa, ga, dc, pc, gc);
  let q1 = edgeVertex(da, pa, ga, dd, pd, gd);
  let q2 = edgeVertex(db, pb, gb, dd, pd, gd);
  let q3 = edgeVertex(db, pb, gb, dc, pc, gc);
  emitTri(q0, q1, q2);
  emitTri(q0, q2, q3);
}

fn emitTet(da: f32, pa: vec3f, ga: vec3f,
           db: f32, pb: vec3f, gb: vec3f,
           dc: f32, pc: vec3f, gc: vec3f,
           dd: f32, pd: vec3f, gd: vec3f) {
  let ia = da > M.iso; let ib = db > M.iso; let ic = dc > M.iso; let id = dd > M.iso;
  var nin = 0;
  if (ia) { nin += 1; } if (ib) { nin += 1; } if (ic) { nin += 1; } if (id) { nin += 1; }
  if (nin == 0 || nin == 4) { return; }
  if (nin == 1 || nin == 3) {
    let look = nin == 1;
    if (ia == look) { emitLone(da,pa,ga, db,pb,gb, dc,pc,gc, dd,pd,gd); }
    else if (ib == look) { emitLone(db,pb,gb, da,pa,ga, dc,pc,gc, dd,pd,gd); }
    else if (ic == look) { emitLone(dc,pc,gc, da,pa,ga, db,pb,gb, dd,pd,gd); }
    else { emitLone(dd,pd,gd, da,pa,ga, db,pb,gb, dc,pc,gc); }
  } else {
    if (ia && ib) { emitQuad(da,pa,ga, db,pb,gb, dc,pc,gc, dd,pd,gd); }
    else if (ia && ic) { emitQuad(da,pa,ga, dc,pc,gc, db,pb,gb, dd,pd,gd); }
    else if (ia && id) { emitQuad(da,pa,ga, dd,pd,gd, db,pb,gb, dc,pc,gc); }
    else if (ib && ic) { emitQuad(db,pb,gb, dc,pc,gc, da,pa,ga, dd,pd,gd); }
    else if (ib && id) { emitQuad(db,pb,gb, dd,pd,gd, da,pa,ga, dc,pc,gc); }
    else { emitQuad(dc,pc,gc, dd,pd,gd, da,pa,ga, db,pb,gb); }
  }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  let nvox = M.vertDim - vec3i(1);
  let total = u32(nvox.x * nvox.y * nvox.z);
  if (idx >= total) { return; }
  var v: vec3i;
  v.x = i32(idx) % nvox.x;
  v.y = (i32(idx) / nvox.x) % nvox.y;
  v.z = i32(idx) / (nvox.x * nvox.y);

  let o0 = vec3i(0,0,0); let o1 = vec3i(1,0,0); let o2 = vec3i(0,1,0); let o3 = vec3i(1,1,0);
  let o4 = vec3i(0,0,1); let o5 = vec3i(1,0,1); let o6 = vec3i(0,1,1); let o7 = vec3i(1,1,1);
  let d0 = D(v + o0); let d1 = D(v + o1); let d2 = D(v + o2); let d3 = D(v + o3);
  let d4 = D(v + o4); let d5 = D(v + o5); let d6 = D(v + o6); let d7 = D(v + o7);
  var cubeMask = 0;
  if (d0 > M.iso) { cubeMask |= 1; }   if (d1 > M.iso) { cubeMask |= 2; }
  if (d2 > M.iso) { cubeMask |= 4; }   if (d3 > M.iso) { cubeMask |= 8; }
  if (d4 > M.iso) { cubeMask |= 16; }  if (d5 > M.iso) { cubeMask |= 32; }
  if (d6 > M.iso) { cubeMask |= 64; }  if (d7 > M.iso) { cubeMask |= 128; }
  if (cubeMask == 0 || cubeMask == 255) { return; }

  let p0 = vec3f(v + o0); let p1 = vec3f(v + o1); let p2 = vec3f(v + o2); let p3 = vec3f(v + o3);
  let p4 = vec3f(v + o4); let p5 = vec3f(v + o5); let p6 = vec3f(v + o6); let p7 = vec3f(v + o7);
  let g0 = gradAt(v + o0); let g1 = gradAt(v + o1);
  let g2 = gradAt(v + o2); let g3 = gradAt(v + o3);
  let g4 = gradAt(v + o4); let g5 = gradAt(v + o5);
  let g6 = gradAt(v + o6); let g7 = gradAt(v + o7);

  emitTet(d0,p0,g0, d1,p1,g1, d3,p3,g3, d7,p7,g7);
  emitTet(d0,p0,g0, d1,p1,g1, d5,p5,g5, d7,p7,g7);
  emitTet(d0,p0,g0, d2,p2,g2, d3,p3,g3, d7,p7,g7);
  emitTet(d0,p0,g0, d2,p2,g2, d6,p6,g6, d7,p7,g7);
  emitTet(d0,p0,g0, d4,p4,g4, d5,p5,g5, d7,p7,g7);
  emitTet(d0,p0,g0, d4,p4,g4, d6,p6,g6, d7,p7,g7);
}
`;

export const meshIndirectWGSL = `
@group(0) @binding(1) var<storage, read>       counter : array<u32>;
@group(0) @binding(2) var<storage, read_write> cmd     : array<u32>;

@compute @workgroup_size(1)
fn main() {
  cmd[0] = min(counter[0], M.maxTris) * 3u;
  cmd[1] = 1u;
  cmd[2] = 0u;
  cmd[3] = 0u;
}
`;

export const meshDrawWGSL = `

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
struct MeshDraw {
  gridOrigin : vec3f,
  voxel      : f32,
  vertDim    : vec3i,
  pad        : f32,
}
@group(0) @binding(0) var<uniform> V : View;
@group(0) @binding(1) var<uniform> G : MeshDraw;
@group(0) @binding(2) var<storage, read> verts : array<vec2u>;

struct VsOut {
  @builtin(position) clip : vec4f,
  @location(0) normal : vec3f,
  @location(1) world  : vec3f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  let data = verts[vi];
  let pn = vec3f(f32(data.x & 0xFFFFu), f32(data.x >> 16u), f32(data.y & 0xFFFFu)) / 65535.0;
  let p = G.gridOrigin + pn * vec3f(G.vertDim - vec3i(1)) * G.voxel;
  let oct = data.y >> 16u;
  let e = vec2f(f32(oct & 0xFFu), f32(oct >> 8u)) / 255.0 * 2.0 - 1.0;
  var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  if (n.z < 0.0) {
    let t = (1.0 - abs(vec2f(n.y, n.x))) *
            vec2f(select(-1.0, 1.0, n.x >= 0.0), select(-1.0, 1.0, n.y >= 0.0));
    n = vec3f(t, n.z);
  }
  var o: VsOut;
  o.normal = normalize(n);
  o.world = p;
  o.clip = V.viewProj * vec4f(p, 1.0);
  return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  var n = normalize(in.normal);
  let viewDir = normalize(V.eye.xyz - in.world);
  if (dot(n, viewDir) < 0.0) { n = -n; }
  let lightDir = normalize(vec3f(0.4, 0.8, 0.6));
  let ndl = max(dot(n, lightDir), 0.0);
  let halfV = normalize(lightDir + viewDir);
  let spec = pow(max(dot(n, halfV), 0.0), 90.0);
  let fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0);
  let deep = vec3f(0.02, 0.15, 0.42);
  let shallow = vec3f(0.12, 0.42, 0.72);
  let sky = vec3f(0.55, 0.65, 0.75);
  var col = mix(deep, shallow, vec3f(ndl));
  col = mix(col, sky, vec3f(0.55 * fresnel));
  col += vec3f(0.9) * spec;
  return vec4f(col, 1.0);
}

struct SurfOut {
  @location(0) eyeZ   : f32,
  @location(1) normal : vec4f,
}

@fragment
fn fsSurf(in: VsOut) -> SurfOut {
  var o : SurfOut;
  o.eyeZ = (V.view * vec4f(in.world, 1.0)).z;

  o.normal = vec4f(normalize(in.normal), 1.0);
  return o;
}
`;
