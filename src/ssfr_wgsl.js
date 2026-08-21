

export const splatPrelude = `
struct Splat {
  view        : mat4x4f,
  proj        : mat4x4f,
  invViewProj : mat4x4f,
  viewport    : vec2f,
  kernelScale : f32,
  quadRadius  : f32,
  thickRadius : f32,
  maxNeighbours : f32,
  skipBodies  : u32,
  pad0        : f32,
}
@group(0) @binding(0) var<uniform> S : Splat;
@group(0) @binding(1) var<storage, read> smoothPos : array<vec4f>;
@group(0) @binding(2) var<storage, read> gdata     : array<vec4f>;
@group(0) @binding(3) var<storage, read> phase     : array<vec4u>;

fn kernelG(i: u32) -> mat3x3f {
  let a = gdata[i * 2u + 0u];
  let b = gdata[i * 2u + 1u];
  return mat3x3f(a.x, a.y, a.z,
                 a.y, a.w, b.x,
                 a.z, b.x, b.y);
}

fn kernelInverse(i: u32) -> mat3x3f {
  let m = kernelG(i);
  let a = m[0][0]; let b = m[1][0]; let c = m[2][0];
  let d = m[0][1]; let e = m[1][1]; let f = m[2][1];
  let g = m[0][2]; let h = m[1][2]; let k = m[2][2];
  let A = e * k - f * h;
  let B = f * g - d * k;
  let C = d * h - e * g;
  let det = a * A + b * B + c * C;
  let inv = 1.0 / select(det, 1.0e-20, abs(det) < 1.0e-20);
  return mat3x3f(A * inv, B * inv, C * inv,
                 (c * h - b * k) * inv, (a * k - c * g) * inv, (b * g - a * h) * inv,
                 (b * f - c * e) * inv, (c * d - a * f) * inv, (a * e - b * d) * inv);
}

struct SplatOut {
  @builtin(position) clip : vec4f,
  @location(0) centre : vec3f,
  @location(1) @interpolate(flat) id : u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SplatOut {
  var o : SplatOut;
  o.id = ii;
  let c = smoothPos[ii].xyz;
  o.centre = c;

  if (S.skipBodies != 0u && phase[ii].x != 0u) {
    o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }

  if (S.maxNeighbours > 0.0 && gdata[ii * 2u + 1u].w > S.maxNeighbours) {
    o.clip = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }

  let ec = S.view * vec4f(c, 1.0);
  let zeye = max(-ec.z, 1.0e-4);
  var clip = S.proj * ec;

  var e : vec3f;
  if (S.quadRadius > 0.0) {
    e = vec3f(S.quadRadius);
  } else {
    let M = mat3x3f(S.view[0].xyz, S.view[1].xyz, S.view[2].xyz)
          * (kernelInverse(ii) * S.kernelScale);
    e = vec3f(length(vec3f(M[0][0], M[1][0], M[2][0])),
              length(vec3f(M[0][1], M[1][1], M[2][1])),
              length(vec3f(M[0][2], M[1][2], M[2][2])));
  }
  let znear = max(zeye - e.z, 1.0e-4);
  let hsize = vec2f(S.proj[0][0], S.proj[1][1]) * e.xy / znear;
  let corner = vec2f(select(-1.0, 1.0, (vi & 1u) != 0u),
                     select(-1.0, 1.0, (vi & 2u) != 0u));
  clip = vec4f(clip.xy + corner * hsize * clip.w, clip.z, clip.w);
  o.clip = clip;
  return o;
}

struct Hit { ok : bool, p : vec3f, chord : f32 }

fn hitEllipsoid(id: u32, centre: vec3f, ro: vec3f, rd: vec3f) -> Hit {
  var h : Hit;
  h.ok = false;
  h.p = vec3f(0.0);
  h.chord = 0.0;

  let G = kernelG(id) * (1.0 / S.kernelScale);
  let oc = ro - centre;
  let ga = G * rd;
  let gb = G * oc;
  let A = dot(ga, ga);
  let B = 2.0 * dot(ga, gb);
  let C = dot(gb, gb) - 1.0;
  let disc = B * B - 4.0 * A * C;
  if (disc <= 0.0 || A <= 0.0) { return h; }
  let sq = sqrt(disc);
  let t = (-B - sq) / (2.0 * A);
  h.chord = sq / A;
  h.p = ro + rd * t;
  h.ok = t > 0.0;
  return h;
}

struct Ray { ro : vec3f, rd : vec3f }

fn fragmentRay(frag: vec2f) -> Ray {
  let ndc = vec2f(frag.x / S.viewport.x * 2.0 - 1.0,
                  1.0 - frag.y / S.viewport.y * 2.0);
  let nearH = S.invViewProj * vec4f(ndc, -1.0, 1.0);
  let farH = S.invViewProj * vec4f(ndc, 1.0, 1.0);
  var r : Ray;
  r.ro = nearH.xyz / nearH.w;
  r.rd = normalize(farH.xyz / farH.w - r.ro);
  return r;
}
`;

export const depthFS = `
struct DepthOut {
  @location(0) eyeZ : f32,
  @builtin(frag_depth) depth : f32,
}

@fragment
fn fs(in: SplatOut) -> DepthOut {
  var o : DepthOut;
  o.depth = 1.0;
  o.eyeZ = -1.0e4;
  let r = fragmentRay(in.clip.xy);
  let h = hitEllipsoid(in.id, in.centre, r.ro, r.rd);
  if (!h.ok) { discard; return o; }
  let clip = S.proj * S.view * vec4f(h.p, 1.0);
  o.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  o.eyeZ = (S.view * vec4f(h.p, 1.0)).z;
  return o;
}
`;

export const thickFS = `
@fragment
fn fs(in: SplatOut) -> @location(0) f32 {
  let r = fragmentRay(in.clip.xy);
  let oc = r.ro - in.centre;
  let b = dot(oc, r.rd);
  let c = dot(oc, oc) - S.thickRadius * S.thickRadius;
  let disc = b * b - c;
  if (disc <= 0.0) { discard; return 0.0; }
  return 2.0 * sqrt(disc);
}
`;

export const thickFilterWGSL = `
struct Blur { dir : vec2i, size : i32, pad : i32 }
@group(0) @binding(0) var<uniform> B : Blur;
@group(0) @binding(1) var src : texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  return vec4f(p, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4f) -> @location(0) f32 {
  let lim = vec2i(textureDimensions(src, 0));
  let ip = vec2i(frag.xy);
  let sigma = max(f32(B.size) / 3.0, 1.0e-3);
  let two_sigma2 = 2.0 * sigma * sigma;
  var sum = 0.0;
  var wsum = 0.0;
  for (var i = -B.size; i <= B.size; i++) {
    let p = clamp(ip + B.dir * i, vec2i(0), lim - vec2i(1));
    let w = exp(-f32(i * i) / two_sigma2);
    sum = sum + textureLoad(src, p, 0).r * w;
    wsum = wsum + w;
  }
  return sum / wsum;
}
`;

export const filterWGSL = `
struct Filt {
  dir         : vec2i,
  maxRadius   : i32,
  mode        : i32,
  sigmaWorld  : f32,
  proj11      : f32,
  viewH       : f32,
  delta       : f32,
  mu          : f32,
  rangeSigma  : f32,
  clean       : i32,
  cleanRadius : i32,
}
@group(0) @binding(0) var<uniform> F : Filt;
@group(0) @binding(1) var eyeZ : texture_2d<f32>;

const kEmptyZ = -1.0e4;
fn isEmpty(z: f32) -> bool { return z < -1.0e3; }

fn fetchZ(p: vec2i, lim: vec2i) -> f32 {
  if (p.x < 0 || p.y < 0 || p.x >= lim.x || p.y >= lim.y) { return kEmptyZ; }
  return textureLoad(eyeZ, p, 0).r;
}

struct NR { z : f32, w : f32, wOther : f32, upper : f32, lower : f32 }

fn narrowRange(zIn: f32, wIn: f32, wOtherIn: f32, upperIn: f32, lowerIn: f32,
               lowerClamp: f32, threshold: f32) -> NR {
  var o : NR;
  o.z = zIn; o.w = wIn; o.wOther = wOtherIn; o.upper = upperIn; o.lower = lowerIn;
  if (o.z > o.upper) {
    o.w = 0.0;
    o.wOther = 0.0;
  } else if (o.z < o.lower) {
    o.z = lowerClamp;
  } else {
    o.upper = max(o.upper, o.z + threshold);
    o.lower = min(o.lower, o.z - threshold);
  }
  return o;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  return vec4f(p, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4f) -> @location(0) f32 {
  let lim = vec2i(textureDimensions(eyeZ, 0));
  let ip = vec2i(frag.xy);
  let zi = textureLoad(eyeZ, ip, 0).r;
  if (isEmpty(zi)) { return zi; }

  let sigma = max(F.sigmaWorld * F.viewH * F.proj11 / (2.0 * abs(zi)), 0.5);
  var R = i32(min(3.0 * sigma, f32(F.maxRadius)));
  if (F.clean != 0) { R = F.cleanRadius; }

  var s = sigma;
  if (F.clean != 0) { s = max(f32(R) / 3.0, 1.0e-3); }
  let inv2s2 = 1.0 / (2.0 * s * s);

  var upperP = zi + F.delta; var lowerP = zi - F.delta;
  var upperM = zi + F.delta; var lowerM = zi - F.delta;
  let lowerClamp = zi - F.mu;
  var sum = zi; var wsum = 1.0;

  if (F.clean != 0) {
    var upper = zi + F.delta; var lower = zi - F.delta;
    for (var dy = 1; dy <= R; dy++) {
      for (var dx = 1; dx <= R; dx++) {
        let g = exp(-f32(dx * dx + dy * dy) * inv2s2);
        var za = fetchZ(ip + vec2i(dx, dy), lim);
        var zb = fetchZ(ip + vec2i(dx, -dy), lim);
        var zc = fetchZ(ip + vec2i(-dx, dy), lim);
        var zd = fetchZ(ip + vec2i(-dx, -dy), lim);
        var wa = g; var wb = g; var wc = g; var wd = g;
        var r = narrowRange(za, wa, wd, upper, lower, lowerClamp, F.delta);
        za = r.z; wa = r.w; wd = r.wOther; upper = r.upper; lower = r.lower;
        r = narrowRange(zb, wb, wc, upper, lower, lowerClamp, F.delta);
        zb = r.z; wb = r.w; wc = r.wOther; upper = r.upper; lower = r.lower;
        r = narrowRange(zc, wc, wb, upper, lower, lowerClamp, F.delta);
        zc = r.z; wc = r.w; wb = r.wOther; upper = r.upper; lower = r.lower;
        r = narrowRange(zd, wd, wa, upper, lower, lowerClamp, F.delta);
        zd = r.z; wd = r.w; wa = r.wOther; upper = r.upper; lower = r.lower;
        sum = sum + za * wa + zb * wb + zc * wc + zd * wd;
        wsum = wsum + wa + wb + wc + wd;
      }
    }

    for (var t = 1; t <= R; t++) {
      let g = exp(-f32(t * t) * inv2s2);
      var zv = array<f32, 4>(fetchZ(ip + vec2i(t, 0), lim),
                             fetchZ(ip + vec2i(-t, 0), lim),
                             fetchZ(ip + vec2i(0, t), lim),
                             fetchZ(ip + vec2i(0, -t), lim));
      var wv = array<f32, 4>(g, g, g, g);
      var r = narrowRange(zv[0], wv[0], wv[1], upper, lower, lowerClamp, F.delta);
      zv[0] = r.z; wv[0] = r.w; wv[1] = r.wOther; upper = r.upper; lower = r.lower;
      r = narrowRange(zv[1], wv[1], wv[0], upper, lower, lowerClamp, F.delta);
      zv[1] = r.z; wv[1] = r.w; wv[0] = r.wOther; upper = r.upper; lower = r.lower;
      r = narrowRange(zv[2], wv[2], wv[3], upper, lower, lowerClamp, F.delta);
      zv[2] = r.z; wv[2] = r.w; wv[3] = r.wOther; upper = r.upper; lower = r.lower;
      r = narrowRange(zv[3], wv[3], wv[2], upper, lower, lowerClamp, F.delta);
      zv[3] = r.z; wv[3] = r.w; wv[2] = r.wOther; upper = r.upper; lower = r.lower;
      for (var i = 0; i < 4; i++) { sum = sum + zv[i] * wv[i]; wsum = wsum + wv[i]; }
    }
  } else {
    for (var t = 1; t <= R; t++) {
      var zj = fetchZ(ip + F.dir * t, lim);
      var zk = fetchZ(ip - F.dir * t, lim);
      let g = exp(-f32(t * t) * inv2s2);
      var wj = g; var wk = g;
      if (F.mode == 2) {
        var r = narrowRange(zj, wj, wk, upperP, lowerP, lowerClamp, F.delta);
        zj = r.z; wj = r.w; wk = r.wOther; upperP = r.upper; lowerP = r.lower;
        r = narrowRange(zk, wk, wj, upperM, lowerM, lowerClamp, F.delta);
        zk = r.z; wk = r.w; wj = r.wOther; upperM = r.upper; lowerM = r.lower;
        sum = sum + zj * wj + zk * wk;
        wsum = wsum + wj + wk;
      } else {
        if (!isEmpty(zj)) {
          var w = g;
          if (F.mode == 1) {
            let dz = zj - zi;
            w = w * exp(-dz * dz / (2.0 * F.rangeSigma * F.rangeSigma));
          }
          sum = sum + w * zj;
          wsum = wsum + w;
        }
        if (!isEmpty(zk)) {
          var w = g;
          if (F.mode == 1) {
            let dz = zk - zi;
            w = w * exp(-dz * dz / (2.0 * F.rangeSigma * F.rangeSigma));
          }
          sum = sum + w * zk;
          wsum = wsum + w;
        }
      }
    }
  }
  return sum / wsum;
}
`;
