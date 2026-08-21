

export const compositePrelude = `
struct Comp {
  invViewProj : mat4x4f,
  invView     : mat4x4f,
  eye         : vec4f,
  boxMin      : vec3f,
  proj00      : f32,
  boxMax      : vec3f,
  proj11      : f32,
  absorb      : vec3f,
  ior         : f32,
  sunDir      : vec3f,
  sunIntensity: f32,
  roughness   : f32,
  exposure    : f32,
  groundReflection : f32,
  thicknessScale : f32,
  bodyCount   : i32,
  floorPlane  : i32,
  debug       : i32,
  hasEnvMap   : i32,
  envIntensity: f32,
  envYaw      : f32,

  mapScale    : vec2f,
}
@group(0) @binding(0) var<uniform> C : Comp;

@group(0) @binding(1) var<storage, read> bdata : array<vec4f>;

@group(0) @binding(7) var uEnvMap : texture_cube<f32>;
@group(0) @binding(8) var uEnvSamp : sampler;

const kHaze = vec3f(0.62, 0.74, 0.90);

fn envSample(d: vec3f) -> vec3f {
  let c = cos(C.envYaw); let s = sin(C.envYaw);
  let r = vec3f(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
  return textureSampleLevel(uEnvMap, uEnvSamp, r, 0.0).rgb * C.envIntensity;
}

fn skyColor(d: vec3f) -> vec3f {
  if (C.hasEnvMap != 0) { return envSample(d); }
  let h = clamp(d.y, 0.0, 1.0);
  let zenith = vec3f(0.09, 0.28, 0.72);
  let mid = vec3f(0.25, 0.48, 0.85);
  var c = mix(kHaze, mid, vec3f(smoothstep(0.0, 0.16, h)));
  c = mix(c, zenith, vec3f(smoothstep(0.10, 0.85, h)));
  let s = max(dot(d, C.sunDir), 0.0);
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
  for (var i = 0; i < C.bodyCount; i++) {
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
  if (C.bodyCount == 0) { return o; }
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
  if (C.floorPlane != 0 && d.y < -1.0e-4) {
    let t = (C.boxMin.y - o.y) / d.y;
    if (t > 0.0) {
      let p = o + d * t;

      let fade = 1.0 - exp(-0.02 * t);
      var far = kHaze;
      if (C.hasEnvMap != 0) { far = envSample(d); }
      return mix(floorColor(p), far, vec3f(fade));
    }
  }
  return skyColor(d);
}

fn shadeBody(p: vec3f, n: vec3f, base: vec3f, rd: vec3f, env: vec3f) -> vec3f {
  let ndl = max(dot(n, C.sunDir), 0.0);
  let h = normalize(C.sunDir - rd);
  let spec = pow(max(dot(n, h), 0.0), 60.0);
  let fres = pow(1.0 - max(dot(-rd, n), 0.0), 4.0);
  return base * (0.22 + 0.78 * ndl) + vec3f(0.7) * spec + env * fres * 0.25;
}

fn envReflect(d: vec3f) -> vec3f {
  if (C.hasEnvMap != 0) { return envSample(d); }
  if (d.y >= 0.0) { return skyColor(d); }
  let k = clamp(-d.y * 4.0, 0.0, 1.0);
  return mix(kHaze, vec3f(0.32, 0.325, 0.335), vec3f(k));
}

fn waterReflect(d: vec3f, n: vec3f) -> vec3f {
  let physical = envReflect(d);
  if (d.y >= 0.0 || C.groundReflection >= 1.0) { return physical; }
  let topSurface = smoothstep(0.819152, 0.939693, n.y);
  let belowHorizon = smoothstep(0.0, 0.10, -d.y);
  let suppress = (1.0 - clamp(C.groundReflection, 0.0, 1.0)) * topSurface * belowHorizon;
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

struct VsOut { @builtin(position) clip : vec4f, @location(0) ndc : vec2f }

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  var o : VsOut;
  o.ndc = p;
  o.clip = vec4f(p, 0.0, 1.0);
  return o;
}
`;

export const bodyDepthFS = `
@fragment
fn fs(in: VsOut) -> @location(0) f32 {
  let nearH = C.invViewProj * vec4f(in.ndc, -1.0, 1.0);
  let farH = C.invViewProj * vec4f(in.ndc, 1.0, 1.0);
  let ro = nearH.xyz / nearH.w;
  let rd = normalize(farH.xyz / farH.w - ro);
  let b = traceBodies(ro, rd, 60.0);
  if (b.t < 0.0) { return 1.0e30; }
  return b.t;
}
`;

export const compositeFS = `
@group(0) @binding(2) var uEyeZ  : texture_2d<f32>;
@group(0) @binding(3) var uRawZ  : texture_2d<f32>;
@group(0) @binding(4) var uThick : texture_2d<f32>;
@group(0) @binding(5) var uBodyT : texture_2d<f32>;
@group(0) @binding(6) var uThickSamp : sampler;

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

fn isEmptyZ(z: f32) -> bool { return z < -1.0e3; }

fn fetchNeighbourZ(p: vec2i, lim: vec2i) -> f32 {
  if (p.x < 0 || p.y < 0 || p.x >= lim.x || p.y >= lim.y) { return -1.0e4; }
  return textureLoad(uEyeZ, p, 0).r;
}

fn viewPos(ndc: vec2f, z: f32) -> vec3f {
  return vec3f(-ndc.x * z / C.proj00, -ndc.y * z / C.proj11, z);
}

fn tonemap(c: vec3f) -> vec3f {
  var col = c * C.exposure;
  col = col / (1.0 + max(max(col.r, col.g), col.b) * 0.35);
  return pow(col, vec3f(1.0 / 2.2));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  let uv = vec2f(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
  let ip = vec2i(in.clip.xy * C.mapScale);
  let lim = vec2i(textureDimensions(uEyeZ, 0));
  let z = textureLoad(uEyeZ, ip, 0).r;
  var thick = textureSampleLevel(uThick, uThickSamp, uv, 0.0).r * C.thicknessScale;

  let nearH = C.invViewProj * vec4f(in.ndc, -1.0, 1.0);
  let farH = C.invViewProj * vec4f(in.ndc, 1.0, 1.0);
  let ro = nearH.xyz / nearH.w;
  let rd = normalize(farH.xyz / farH.w - ro);

  if (C.debug == 3) { return vec4f(thick, thick * 0.25, 0.0, 1.0); }
  if (C.debug == 2 || C.debug == 4) {
    var d = z;
    if (C.debug == 4) { d = textureLoad(uRawZ, ip, 0).r; }
    var s = 0.0;
    if (!isEmptyZ(d)) { s = clamp(-d / 4.0, 0.0, 1.0); }
    return vec4f(vec3f(1.0 - s), 1.0);
  }

  let tBody = textureLoad(uBodyT, ip, 0).r;

  if (C.debug == 5) {
    if (tBody > 1.0e29) { return vec4f(0.0, 0.0, 0.6, 1.0); }
    return vec4f(fract(tBody), fract(tBody * 0.1), 0.0, 1.0);
  }

  let p0 = viewPos(in.ndc, z);
  let p = (C.invView * vec4f(p0, 1.0)).xyz;
  let bodyInFront = tBody < length(p - ro);

  if (isEmptyZ(z) || bodyInFront) {
    return vec4f(tonemap(sceneColor(ro, rd)), 1.0);
  }

  let px = 2.0 / vec2f(lim);
  let zxp = fetchNeighbourZ(ip + vec2i(1, 0), lim);
  let zxm = fetchNeighbourZ(ip - vec2i(1, 0), lim);

  let zyp = fetchNeighbourZ(ip - vec2i(0, 1), lim);
  let zym = fetchNeighbourZ(ip + vec2i(0, 1), lim);
  let dxr = viewPos(in.ndc + vec2f(px.x, 0.0), zxp) - p0;
  let dxl = p0 - viewPos(in.ndc - vec2f(px.x, 0.0), zxm);
  let dyt = viewPos(in.ndc + vec2f(0.0, px.y), zyp) - p0;
  let dyb = p0 - viewPos(in.ndc - vec2f(0.0, px.y), zym);
  var dpx = dxl;
  if (abs(dxr.z) < abs(dxl.z)) { dpx = dxr; }
  var dpy = dyb;
  if (abs(dyt.z) < abs(dyb.z)) { dpy = dyt; }
  let nView = normalize(cross(dpx, dpy));
  let iv = mat3x3f(C.invView[0].xyz, C.invView[1].xyz, C.invView[2].xyz);
  var n = normalize(iv * nView);
  if (any(n != n)) { n = -rd; }
  if (dot(n, rd) > 0.0) { n = -n; }

  if (C.debug == 1) { return vec4f(n * 0.5 + 0.5, 1.0); }

  let kS = fresnelFull(max(dot(-rd, n), 0.0), 1.0, C.ior);
  let refl = waterReflect(reflect(rd, n), n);
  var refrDir = refract(rd, n, 1.0 / C.ior);
  if (dot(refrDir, refrDir) < 1.0e-8) { refrDir = reflect(rd, n); }

  let ro2 = p + refrDir * 1.0e-3;
  let st = traceBodies(ro2, refrDir, 60.0);
  var hitCol : vec3f;
  if (st.t >= 0.0) {
    let sp = ro2 + refrDir * st.t;
    hitCol = shadeBody(sp, st.nrm, st.col, refrDir, envReflect(reflect(refrDir, st.nrm)));
    thick = min(thick, st.t);
  } else {
    hitCol = background(ro2, refrDir);
  }
  let trans = hitCol * exp(-C.absorb * thick);

  var col = mix(trans, refl, vec3f(kS));
  col += vec3f(1.0, 0.97, 0.90) * C.sunIntensity *
         ggxSpec(n, -rd, C.sunDir, C.roughness) * kS;
  return vec4f(tonemap(col), 1.0);
}
`;
