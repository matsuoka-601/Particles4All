

export const anisoWGSL = `
struct AnisoParams {
  count      : u32,
  lambda     : f32,
  invSupport : f32,
  support    : f32,
  boxMin     : vec3f,
  radius     : f32,
  simGridDim : vec3i,
  cells      : i32,
  kr         : f32,
  kn         : f32,
  ks         : f32,
  stretch    : f32,
  minExtent  : f32,
  neps       : i32,
  pad0       : f32,
  pad1       : f32,
}
@group(0) @binding(0) var<uniform> A : AnisoParams;
@group(0) @binding(1) var<storage, read>       pos       : array<vec4f>;
@group(0) @binding(2) var<storage, read>       cellStart : array<u32>;
@group(0) @binding(3) var<storage, read_write> smoothPos : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> g         : array<vec4f>;

struct Eigen { a : mat3x3f, v : mat3x3f }

fn jacobi(aIn: mat3x3f) -> Eigen {
  var a = aIn;
  var v = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
  for (var sweep = 0; sweep < 6; sweep++) {
    for (var p = 0; p < 2; p++) {
      for (var q = p + 1; q < 3; q++) {
        let apq = a[q][p];
        if (abs(apq) < 1.0e-12) { continue; }
        let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
        var t = sign(theta) / (abs(theta) + sqrt(theta * theta + 1.0));
        if (theta == 0.0) { t = 1.0; }
        let cth = 1.0 / sqrt(t * t + 1.0);
        let s = t * cth;
        var r = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        r[p][p] = cth; r[q][q] = cth;
        r[q][p] = s;   r[p][q] = -s;
        a = transpose(r) * a * r;
        v = v * r;
      }
    }
  }
  var e : Eigen;
  e.a = a;
  e.v = v;
  return e;
}

fn diag(s: f32) -> mat3x3f {
  return mat3x3f(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s);
}

fn storeG(i: u32, G: mat3x3f) {
  g[i * 2u + 0u] = vec4f(G[0][0], G[1][0], G[2][0], G[1][1]);
  g[i * 2u + 1u] = vec4f(G[2][1], G[2][2], determinant(G), 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
  let i = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  if (i >= A.count) { return; }
  let xi = pos[i].xyz;
  let c = clamp(vec3i(floor((xi - A.boxMin) * A.invSupport)),
                vec3i(0), A.simGridDim - vec3i(1));
  let x0 = max(c.x - A.cells, 0);
  let x1 = min(c.x + A.cells, A.simGridDim.x - 1);

  let r2max = A.radius * A.radius;
  let invR = 1.0 / A.radius;

  let rel = xi - A.boxMin;
  var sd = vec3f(0.0);
  var m2 = mat3x3f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  var wsum = 0.0;
  var n = 0;
  for (var dz = -A.cells; dz <= A.cells; dz++) {
    let z = c.z + dz;
    if (z < 0 || z >= A.simGridDim.z) { continue; }
    let dzMin = max(max(f32(z) * A.support - rel.z,
                        rel.z - f32(z + 1) * A.support), 0.0);
    let dz2 = dzMin * dzMin;
    if (dz2 >= r2max) { continue; }
    for (var dy = -A.cells; dy <= A.cells; dy++) {
      let y = c.y + dy;
      if (y < 0 || y >= A.simGridDim.y) { continue; }
      let dyMin = max(max(f32(y) * A.support - rel.y,
                          rel.y - f32(y + 1) * A.support), 0.0);
      if (dz2 + dyMin * dyMin >= r2max) { continue; }
      let rowBase = u32((z * A.simGridDim.y + y) * A.simGridDim.x);
      let jb = cellStart[rowBase + u32(x0)];
      let je = cellStart[rowBase + u32(x1) + 1u];
      for (var j = jb; j < je; j++) {
        let d = pos[j].xyz - xi;
        let r2 = dot(d, d);

        if (r2 >= r2max) { continue; }

        let t = sqrt(r2) * invR;
        let w = 1.0 - t * t * t;
        sd = sd + d * w;
        m2[0] = m2[0] + d * (d.x * w);
        m2[1] = m2[1] + d * (d.y * w);
        m2[2] = m2[2] + d * (d.z * w);
        wsum = wsum + w;
        n = n + 1;
      }
    }
  }
  if (wsum <= 0.0) {
    smoothPos[i] = vec4f(xi, 0.0);
    storeG(i, diag(A.kn / A.support));
    return;
  }

  let dbar = sd / wsum;
  smoothPos[i] = vec4f(xi + dbar * A.lambda, 0.0);

  var cov = m2;
  cov[0] = cov[0] / wsum - dbar * dbar.x;
  cov[1] = cov[1] / wsum - dbar * dbar.y;
  cov[2] = cov[2] / wsum - dbar * dbar.z;

  let e = jacobi(cov);
  let R = e.v;
  let sig = vec3f(e.a[0][0], e.a[1][1], e.a[2][2]);
  let s1 = max(sig.x, max(sig.y, sig.z));

  let sigT = max(sig, vec3f(s1 / A.kr));
  var ext = A.ks * sigT;

  ext = ext / max(1.0, max(ext.x, max(ext.y, ext.z)) / A.stretch);

  ext = max(ext, vec3f(A.minExtent));

  let inv = 1.0 / (A.support * ext);
  let Gan = R * mat3x3f(inv.x, 0.0, 0.0, 0.0, inv.y, 0.0, 0.0, 0.0, inv.z)
              * transpose(R);

  let Giso = diag(A.kn / A.support);
  let w = smoothstep(f32(A.neps) * 0.4, f32(A.neps) * 1.6, f32(n));
  var G : mat3x3f;
  G[0] = mix(Giso[0], Gan[0], vec3f(w));
  G[1] = mix(Giso[1], Gan[1], vec3f(w));
  G[2] = mix(Giso[2], Gan[2], vec3f(w));

  g[i * 2u + 0u] = vec4f(G[0][0], G[1][0], G[2][0], G[1][1]);

  g[i * 2u + 1u] = vec4f(G[2][1], G[2][2], determinant(G), f32(n));
}
`;
