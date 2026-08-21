

export const rayWGSL = `
struct RayParams {
  invViewProj : mat4x4f,
  invView     : mat4x4f,
  eye         : vec4f,
  gridOrigin  : vec3f,
  voxel       : f32,
  boxMin      : vec3f,
  iso         : f32,
  boxMax      : vec3f,
  ior         : f32,
  absorb      : vec3f,
  skin        : f32,
  sunDir      : vec3f,
  sunIntensity: f32,
  vertDim     : vec3i,
  thickSteps  : i32,
  roughness   : f32,
  exposure    : f32,
  groundReflection : f32,
  bodyCount   : i32,
  floorPlane  : i32,
  debug       : i32,
  useSurface  : i32,
  proj00      : f32,
  proj11      : f32,
  hasEnvMap   : i32,
  envIntensity: f32,
  envYaw      : f32,
}
@group(0) @binding(0) var<uniform> R : RayParams;
@group(0) @binding(1) var<storage, read> field  : array<f32>;

@group(0) @binding(2) var<storage, read> bdata  : array<vec4f>;

@group(0) @binding(3) var surfZ : texture_2d<f32>;
@group(0) @binding(4) var surfN : texture_2d<f32>;

@group(0) @binding(5) var uEnvMap : texture_cube<f32>;
@group(0) @binding(6) var uEnvSamp : sampler;

const kHaze = vec3f(0.62, 0.74, 0.90);

fn envSample(d: vec3f) -> vec3f {
  let c = cos(R.envYaw); let s = sin(R.envYaw);
  let r = vec3f(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
  return textureSampleLevel(uEnvMap, uEnvSamp, r, 0.0).rgb * R.envIntensity;
}

fn F(cc: vec3i) -> f32 {
  let c = clamp(cc, vec3i(0), R.vertDim - vec3i(1));
  return field[u32((c.z * R.vertDim.y + c.y) * R.vertDim.x + c.x)];
}

fn skyColor(d: vec3f) -> vec3f {
  if (R.hasEnvMap != 0) { return envSample(d); }
  let h = clamp(d.y, 0.0, 1.0);
  let zenith = vec3f(0.09, 0.28, 0.72);
  let mid = vec3f(0.25, 0.48, 0.85);
  var c = mix(kHaze, mid, vec3f(smoothstep(0.0, 0.16, h)));
  c = mix(c, zenith, vec3f(smoothstep(0.10, 0.85, h)));
  let s = max(dot(d, R.sunDir), 0.0);
  c += vec3f(1.00, 0.96, 0.88) * pow(s, 3000.0) * 14.0;
  c += vec3f(1.00, 0.92, 0.78) * pow(s, 24.0) * 0.20;
  return c;
}

fn floorColor(p: vec3f) -> vec3f {
  let base = vec3f(0.30, 0.305, 0.315);
  let g = abs(fract(p.xz) - vec2f(0.5));
  let line = 1.0 - smoothstep(0.0, 0.015, min(g.x, g.y));
  var c = mix(base, vec3f(0.50, 0.51, 0.52), vec3f(line * 0.8));
  let chk = (floor(p.x) + floor(p.z)) - 2.0 * floor((floor(p.x) + floor(p.z)) * 0.5);
  c *= mix(0.88, 1.10, chk);
  return c;
}

fn bodyDist(i: i32, p: vec3f) -> f32 {
  let b = u32(i * 6);
  let centre = bdata[b].xyz;
  let shape = i32(bdata[b].w + 0.5);
  let h = bdata[b + 1u].xyz;
  let d0 = p - centre;
  let q = vec3f(dot(bdata[b + 2u].xyz, d0), dot(bdata[b + 3u].xyz, d0),
                dot(bdata[b + 4u].xyz, d0));
  if (shape == 1) { return length(q) - h.x; }
  if (shape == 2) {
    let t = vec2f(length(q.xz) - h.x, q.y);
    return length(t) - h.y;
  }
  let d = abs(q) - h;
  return length(max(d, vec3f(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

struct SceneHit { dist : f32, id : i32 }

fn bodyScene(p: vec3f) -> SceneHit {
  var o : SceneHit;
  o.dist = 1.0e30;
  o.id = -1;
  for (var i = 0; i < R.bodyCount; i++) {
    let d = bodyDist(i, p);
    if (d < o.dist) { o.dist = d; o.id = i; }
  }
  return o;
}

fn bodyNormal(i: i32, p: vec3f) -> vec3f {
  let e = 2.0e-4;
  return normalize(vec3f(
    bodyDist(i, p + vec3f(e,0,0)) - bodyDist(i, p - vec3f(e,0,0)),
    bodyDist(i, p + vec3f(0,e,0)) - bodyDist(i, p - vec3f(0,e,0)),
    bodyDist(i, p + vec3f(0,0,e)) - bodyDist(i, p - vec3f(0,0,e))));
}

struct BodyTrace { t : f32, nrm : vec3f, col : vec3f }

fn traceBodies(ro: vec3f, rd: vec3f, tMax: f32) -> BodyTrace {
  var o : BodyTrace;
  o.t = -1.0;
  o.nrm = vec3f(0.0, 1.0, 0.0);
  o.col = vec3f(0.8);
  if (R.bodyCount == 0) { return o; }
  var t = 1.0e-4;
  for (var i = 0; i < 128; i++) {
    let p = ro + rd * t;
    let s = bodyScene(p);
    if (s.dist < 5.0e-5) {
      o.nrm = bodyNormal(s.id, p);
      o.col = bdata[u32(s.id * 6 + 5)].xyz;
      o.t = t;
      return o;
    }
    t += max(s.dist, 1.0e-4);
    if (t > tMax) { break; }
  }
  return o;
}

fn background(o: vec3f, d: vec3f) -> vec3f {
  if (R.floorPlane != 0 && d.y < -1.0e-4) {
    let t = (R.boxMin.y - o.y) / d.y;
    if (t > 0.0) {
      let p = o + d * t;

      let fade = 1.0 - exp(-0.02 * t);
      var far = kHaze;
      if (R.hasEnvMap != 0) { far = envSample(d); }
      return mix(floorColor(p), far, vec3f(fade));
    }
  }
  return skyColor(d);
}

fn shadeBody(p: vec3f, n: vec3f, base: vec3f, rd: vec3f, env: vec3f) -> vec3f {
  let ndl = max(dot(n, R.sunDir), 0.0);
  let h = normalize(R.sunDir - rd);
  let spec = pow(max(dot(n, h), 0.0), 60.0);
  let fres = pow(1.0 - max(dot(-rd, n), 0.0), 4.0);
  return base * (0.22 + 0.78 * ndl) + vec3f(0.7) * spec + env * fres * 0.25;
}

fn envReflect(d: vec3f) -> vec3f {
  if (R.hasEnvMap != 0) { return envSample(d); }
  if (d.y >= 0.0) { return skyColor(d); }
  let k = clamp(-d.y * 4.0, 0.0, 1.0);
  return mix(kHaze, vec3f(0.32, 0.325, 0.335), vec3f(k));
}

fn waterReflect(d: vec3f, n: vec3f) -> vec3f {
  let physical = envReflect(d);
  if (d.y >= 0.0 || R.groundReflection >= 1.0) { return physical; }
  let topSurface = smoothstep(0.819152, 0.939693, n.y);
  let belowHorizon = smoothstep(0.0, 0.10, -d.y);
  let suppress = (1.0 - clamp(R.groundReflection, 0.0, 1.0)) * topSurface * belowHorizon;
  let xz = d.xz;
  let xzLength = max(length(xz), 1.0e-4);
  let horizonDir = vec3f(xz.x / xzLength, 0.0, xz.y / xzLength);
  return mix(physical, skyColor(horizonDir), vec3f(suppress));
}

fn sceneColor(o: vec3f, d: vec3f) -> vec3f {
  let b = traceBodies(o, d, 60.0);
  if (b.t < 0.0) { return background(o, d); }
  let p = o + d * b.t;
  return shadeBody(p, b.nrm, b.col, d, background(p + b.nrm * 1.0e-4, reflect(d, b.nrm)));
}

struct Cell { d000:f32, d100:f32, d010:f32, d110:f32, d001:f32, d101:f32, d011:f32, d111:f32 }

fn cellAt(c: vec3i) -> Cell {
  var k : Cell;
  k.d000 = F(c + vec3i(0,0,0)); k.d100 = F(c + vec3i(1,0,0));
  k.d010 = F(c + vec3i(0,1,0)); k.d110 = F(c + vec3i(1,1,0));
  k.d001 = F(c + vec3i(0,0,1)); k.d101 = F(c + vec3i(1,0,1));
  k.d011 = F(c + vec3i(0,1,1)); k.d111 = F(c + vec3i(1,1,1));
  return k;
}

fn triLocal(k: Cell, u: vec3f) -> f32 {
  let x00 = mix(k.d000, k.d100, u.x); let x10 = mix(k.d010, k.d110, u.x);
  let x01 = mix(k.d001, k.d101, u.x); let x11 = mix(k.d011, k.d111, u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

fn triAt(g: vec3f) -> f32 {
  let c = vec3i(floor(g));
  return triLocal(cellAt(c), clamp(g - vec3f(c), vec3f(0.0), vec3f(1.0)));
}

fn smoothNormal(g: vec3f) -> vec3f {
  let h = 1.0;
  return vec3f(triAt(g + vec3f(h,0,0)) - triAt(g - vec3f(h,0,0)),
               triAt(g + vec3f(0,h,0)) - triAt(g - vec3f(0,h,0)),
               triAt(g + vec3f(0,0,h)) - triAt(g - vec3f(0,0,h)));
}

fn gradLocal(k: Cell, u: vec3f) -> vec3f {
  let ox = 1.0 - u.x; let oy = 1.0 - u.y; let oz = 1.0 - u.z;
  let gx = (k.d100 - k.d000) * oy * oz + (k.d110 - k.d010) * u.y * oz +
           (k.d101 - k.d001) * oy * u.z + (k.d111 - k.d011) * u.y * u.z;
  let gy = (k.d010 - k.d000) * ox * oz + (k.d110 - k.d100) * u.x * oz +
           (k.d011 - k.d001) * ox * u.z + (k.d111 - k.d101) * u.x * u.z;
  let gz = (k.d001 - k.d000) * ox * oy + (k.d101 - k.d100) * u.x * oy +
           (k.d011 - k.d010) * ox * u.y + (k.d111 - k.d110) * u.x * u.y;
  return vec3f(gx, gy, gz);
}

struct Cubic { A:f32, B:f32, C:f32, D:f32 }

fn cubicOf(k: Cell, u0: vec3f, du: vec3f) -> Cubic {
  let f0 = triLocal(k, u0);
  let f1 = triLocal(k, u0 + du * (1.0 / 3.0));
  let f2 = triLocal(k, u0 + du * (2.0 / 3.0));
  let f3 = triLocal(k, u0 + du);
  let g1 = f1 - f0; let g2 = f2 - f0; let g3 = f3 - f0;
  var c : Cubic;
  c.A = 4.5 * (3.0 * g1 - 3.0 * g2 + g3);
  c.B = -(27.0 * g1 - 9.0 * g3 + 8.0 * c.A) / 6.0;
  c.C = g3 - c.A - c.B;
  c.D = f0;
  return c;
}

fn cubicAt(c: Cubic, t: f32) -> f32 {
  return ((c.A * t + c.B) * t + c.C) * t + c.D;
}

fn solveMonotone(c: Cubic, loIn: f32, hiIn: f32, vloIn: f32, vhiIn: f32) -> f32 {
  var lo = loIn; var hi = hiIn; var vlo = vloIn; var vhi = vhiIn;
  var t = lo;
  for (var i = 0; i < 12; i++) {
    t = lo + (hi - lo) * (-vlo) / max(vhi - vlo, 1.0e-9);
    t = clamp(t, lo, hi);
    let v = cubicAt(c, t) - R.iso;
    if (v * vlo > 0.0) { lo = t; vlo = v; } else { hi = t; vhi = v; }
  }
  return t;
}

fn segmentCrossing(k: Cell, u0: vec3f, du: vec3f) -> f32 {
  let c = cubicOf(k, u0, du);

  var e0 = 2.0; var e1 = 2.0;
  let a = 3.0 * c.A; let b = 2.0 * c.B; let cc = c.C;
  if (abs(a) > 1.0e-12) {
    let disc = b * b - 4.0 * a * cc;
    if (disc > 0.0) {
      let sq = sqrt(disc);
      let r0 = (-b - sq) / (2.0 * a);
      let r1 = (-b + sq) / (2.0 * a);
      e0 = min(r0, r1);
      e1 = max(r0, r1);
    }
  } else if (abs(b) > 1.0e-12) {
    e0 = -cc / b;
  }
  var bounds = array<f32, 4>(0.0, 1.0, 1.0, 1.0);
  var n = 1;
  if (e0 > 0.0 && e0 < 1.0) { bounds[n] = e0; n += 1; }
  if (e1 > 0.0 && e1 < 1.0 && e1 > e0) { bounds[n] = e1; n += 1; }
  bounds[n] = 1.0; n += 1;

  var lo = bounds[0];
  var vlo = cubicAt(c, lo) - R.iso;
  for (var i = 1; i < n; i++) {
    let hi = bounds[i];
    let vhi = cubicAt(c, hi) - R.iso;
    if ((vlo <= 0.0 && vhi >= 0.0) || (vlo >= 0.0 && vhi <= 0.0)) {
      return solveMonotone(c, lo, hi, vlo, vhi);
    }
    lo = hi;
    vlo = vhi;
  }
  return -1.0;
}

struct Slab { hit : bool, t0 : f32, t1 : f32 }

fn boxHit(o: vec3f, d: vec3f, lo: vec3f, hi: vec3f) -> Slab {
  let inv = 1.0 / d;
  let a = (lo - o) * inv;
  let b = (hi - o) * inv;
  let tmin = min(a, b);
  let tmax = max(a, b);
  var s : Slab;
  s.t0 = max(max(tmin.x, tmin.y), tmin.z);
  s.t1 = min(min(tmax.x, tmax.y), tmax.z);
  s.hit = s.t1 > max(s.t0, 0.0);
  return s;
}

struct March { t : f32, pos : vec3f, nrm : vec3f }

fn march(ro: vec3f, rd: vec3f, tStart: f32, tEnd: f32) -> March {
  var out : March;
  out.t = -1.0;
  out.pos = vec3f(0.0);
  out.nrm = vec3f(0.0, 1.0, 0.0);
  let g0 = (ro - R.gridOrigin) / R.voxel;
  let gd = rd / R.voxel;

  let p0 = g0 + gd * tStart;
  var c = clamp(vec3i(floor(p0)), vec3i(0), R.vertDim - vec3i(2));
  let stp = vec3i(sign(gd));
  let invD = 1.0 / max(abs(gd), vec3f(1.0e-12));
  let nextEdge = vec3f(c) + max(vec3f(stp), vec3f(0.0));
  var tMax = (nextEdge - g0) / gd;
  var tDelta = invD;
  if (gd.x == 0.0) { tMax.x = 1.0e30; tDelta.x = 1.0e30; }
  if (gd.y == 0.0) { tMax.y = 1.0e30; tDelta.y = 1.0e30; }
  if (gd.z == 0.0) { tMax.z = 1.0e30; tDelta.z = 1.0e30; }

  var t = tStart;
  for (var i = 0; i < 4096; i++) {
    let tNext = min(min(tMax.x, tMax.y), tMax.z);
    let segEnd = min(tNext, tEnd);
    var stepped = false;
    if (segEnd > t) {
      let k = cellAt(c);
      let u0 = g0 + gd * t - vec3f(c);
      let du = gd * (segEnd - t);
      let s = segmentCrossing(k, u0, du);
      if (s >= 0.0) {
        let tHit = t + s * (segEnd - t);
        let p = ro + rd * tHit;

        var inside = false;
        if (R.bodyCount > 0) { inside = bodyScene(p).dist < 0.0; }
        if (!inside) {
          out.t = tHit;
          out.pos = p;
          out.nrm = gradLocal(k, clamp(u0 + du * s, vec3f(0.0), vec3f(1.0)));
          return out;
        }
      }
    }
    t = tNext;
    if (t >= tEnd) { break; }
    if (tMax.x <= tMax.y && tMax.x <= tMax.z) { c.x += stp.x; tMax.x += tDelta.x; }
    else if (tMax.y <= tMax.z) { c.y += stp.y; tMax.y += tDelta.y; }
    else { c.z += stp.z; tMax.z += tDelta.z; }
    if (any(c < vec3i(0)) || any(c > R.vertDim - vec3i(2))) { break; }
  }
  return out;
}

fn fresnelFull(cosI: f32, n1: f32, n2: f32) -> f32 {
  let eta = n1 / n2;
  let sinT2 = eta * eta * (1.0 - cosI * cosI);
  if (sinT2 >= 1.0) { return 1.0; }
  let cosT = sqrt(1.0 - sinT2);
  let rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
  let rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
  return clamp(0.5 * (rs * rs + rp * rp), 0.0, 1.0);
}

fn ggxSpec(n: vec3f, v: vec3f, l: vec3f, rough: f32) -> f32 {
  let a = max(rough * rough, 1.0e-4);
  let h = normalize(l + v);
  let ndh = max(dot(n, h), 0.0);
  let ndl = max(dot(n, l), 0.0);
  let ndv = max(dot(n, v), 0.0);
  let a2 = a * a;
  let d = ndh * ndh * (a2 - 1.0) + 1.0;
  let ndf = a2 / max(3.14159265 * d * d, 1.0e-7);
  let k = a * 0.5;
  let gl = ndl / max(ndl * (1.0 - k) + k, 1.0e-5);
  let gv = ndv / max(ndv * (1.0 - k) + k, 1.0e-5);
  return ndf * gl * gv * ndl / max(4.0 * ndl * ndv, 1.0e-4);
}

struct MeshHit { hit : bool, world : vec3f, normal : vec3f }

fn meshSurfaceHit(ndc: vec2f, frag: vec2f) -> MeshHit {
  var o : MeshHit;
  o.hit = false;
  o.world = vec3f(0.0);
  o.normal = vec3f(0.0, 1.0, 0.0);
  let ip = vec2i(frag);
  let z = textureLoad(surfZ, ip, 0).r;
  if (z < -1.0e3) { return o; }
  let pView = vec3f(-ndc.x * z / R.proj00, -ndc.y * z / R.proj11, z);
  o.world = (R.invView * vec4f(pView, 1.0)).xyz;
  let n = textureLoad(surfN, ip, 0).xyz;
  if (dot(n, n) < 1.0e-8) { return o; }
  o.normal = normalize(n);
  o.hit = true;
  return o;
}

fn opticalDepth(ro: vec3f, rd: vec3f, tMax: f32) -> f32 {
  let volLo = R.gridOrigin;
  let volHi = R.gridOrigin + R.voxel * vec3f(R.vertDim - vec3i(1));
  let s = boxHit(ro, rd, volLo, volHi);
  if (!s.hit) { return 0.0; }
  let a = max(s.t0, 0.0);
  let b = min(max(s.t1, 0.0), tMax);
  if (b <= a) { return 0.0; }
  let dt = (b - a) / f32(R.thickSteps);
  var acc = 0.0;
  for (var i = 0; i < R.thickSteps; i++) {
    let t = a + (f32(i) + 0.5) * dt;
    let f = triAt((ro + rd * t - R.gridOrigin) / R.voxel);
    acc += smoothstep(R.iso * 0.55, R.iso * 1.45, f);
  }
  return acc * dt;
}

struct VsOut { @builtin(position) clip : vec4f, @location(0) ndc : vec2f }

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  var o : VsOut;
  o.ndc = p;
  o.clip = vec4f(p, 0.0, 1.0);
  return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  let nearH = R.invViewProj * vec4f(in.ndc, -1.0, 1.0);
  let farH = R.invViewProj * vec4f(in.ndc, 1.0, 1.0);
  let nearP = nearH.xyz / nearH.w;
  let farP = farH.xyz / farH.w;
  let ro = R.eye.xyz;
  let rd = normalize(farP - nearP);

  let volLo = R.gridOrigin;
  let volHi = R.gridOrigin + R.voxel * vec3f(R.vertDim - vec3i(1));

  var tHit = -1.0;
  var hit = vec3f(0.0);
  var gn = vec3f(0.0, 1.0, 0.0);
  var meshN = vec3f(0.0, 1.0, 0.0);
  if (R.useSurface != 0) {
    let mh = meshSurfaceHit(in.ndc, in.clip.xy);
    if (mh.hit) {
      hit = mh.world;
      meshN = mh.normal;
      tHit = length(hit - ro);
    }
  } else {
    let slab = boxHit(ro, rd, volLo, volHi);
    if (slab.hit) {
      let m = march(ro, rd, max(slab.t0, 0.0) + 1.0e-5, slab.t1);
      tHit = m.t;
      hit = m.pos;
      gn = m.nrm;
    }
  }

  var tMaxBody = 60.0;
  if (tHit >= 0.0) { tMaxBody = tHit + R.skin; }
  let bt = traceBodies(ro, rd, tMaxBody);

  var col = vec3f(0.0);
  var n = vec3f(0.0, 1.0, 0.0);
  var thick = 0.0;

  if (bt.t >= 0.0 && (tHit < 0.0 || bt.t < tHit + R.skin)) {
    let p = ro + rd * bt.t;
    col = shadeBody(p, bt.nrm, bt.col, rd, envReflect(reflect(rd, bt.nrm)));
  } else if (tHit < 0.0) {
    col = background(ro, rd);
  } else {
    if (R.useSurface != 0) { n = meshN; }
    else { n = normalize(-smoothNormal((hit - R.gridOrigin) / R.voxel)); }
    if (dot(n, rd) > 0.0) { n = -n; }

    let kS = fresnelFull(max(dot(-rd, n), 0.0), 1.0, R.ior);
    let reflDir = reflect(rd, n);
    var refrDir = refract(rd, n, 1.0 / R.ior);
    if (dot(refrDir, refrDir) < 1.0e-8) { refrDir = reflDir; }

    let refl = waterReflect(reflDir, n);

    let ro2 = hit + refrDir * (R.voxel * 0.02);
    let st = traceBodies(ro2, refrDir, 60.0);
    var trans = vec3f(0.0);
    if (st.t >= 0.0) {
      let p = ro2 + refrDir * st.t;
      trans = shadeBody(p, st.nrm, st.col, refrDir, envReflect(reflect(refrDir, st.nrm)));
    } else {
      trans = background(ro2, refrDir);
    }
    var tSolid = 1.0e30;
    if (st.t >= 0.0) { tSolid = st.t; }
    thick = opticalDepth(ro2, refrDir, tSolid);
    trans *= exp(-R.absorb * thick);

    col = mix(trans, refl, vec3f(kS));
    col += vec3f(1.0, 0.97, 0.90) * R.sunIntensity *
           ggxSpec(n, -rd, R.sunDir, R.roughness) * kS;
  }

  if (R.debug == 1) {
    col = select(n * 0.5 + 0.5, vec3f(0.0), tHit < 0.0);
  } else if (R.debug == 2) {
    col = select(vec3f(fract(tHit), fract(tHit * 0.1), 0.0), vec3f(0.0), tHit < 0.0);
  } else if (R.debug == 3) {
    col = vec3f(thick, thick * 0.25, 0.0);
  } else if (R.debug == 4) {
    var c4 = vec3f(0.1, 0.3, 1.0);
    if (reflect(rd, n).y < 0.0) { c4 = vec3f(1.0, 0.1, 0.1); }
    col = select(c4, vec3f(0.0), tHit < 0.0);
  } else if (R.debug == 6) {
    col = select(waterReflect(reflect(rd, n), n), background(ro, rd), tHit < 0.0);
  } else if (R.debug == 5) {
    if (tHit < 0.0) { col = vec3f(0.0); }
    else {
      let deg = degrees(acos(clamp(abs(n.y), 0.0, 1.0)));
      if (deg < 5.0) { col = vec3f(0.1, 0.3, 1.0); }
      else if (deg < 10.0) { col = vec3f(0.1, 0.9, 0.9); }
      else if (deg < 20.0) { col = vec3f(0.2, 0.9, 0.2); }
      else if (deg < 40.0) { col = vec3f(0.95, 0.9, 0.2); }
      else { col = vec3f(1.0, 0.15, 0.1); }
    }
  }

  col *= R.exposure;
  col = col / (1.0 + max(max(col.r, col.g), col.b) * 0.35);
  col = pow(col, vec3f(1.0 / 2.2));
  return vec4f(col, 1.0);
}
`;
