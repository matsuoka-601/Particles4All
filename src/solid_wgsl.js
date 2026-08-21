

export const bodyPackWGSL = `
struct Pack { count : u32 }
@group(0) @binding(0) var<uniform> P : Pack;
@group(0) @binding(1) var<storage, read>       centre : array<vec4f>;
@group(0) @binding(2) var<storage, read>       rot    : array<vec4f>;
@group(0) @binding(3) var<storage, read>       stat   : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> out    : array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.count) { return; }
  let b = i * 6u;

  let s = stat[i * 2u];

  let r0 = rot[i * 3u + 0u].xyz;
  let r1 = rot[i * 3u + 1u].xyz;
  let r2 = rot[i * 3u + 2u].xyz;
  out[b + 0u] = vec4f(centre[i].xyz, s.w);
  out[b + 1u] = vec4f(s.xyz, 0.0);
  out[b + 2u] = vec4f(r0.x, r1.x, r2.x, 0.0);
  out[b + 3u] = vec4f(r0.y, r1.y, r2.y, 0.0);
  out[b + 4u] = vec4f(r0.z, r1.z, r2.z, 0.0);
  out[b + 5u] = vec4f(stat[i * 2u + 1u].xyz, 0.0);
}
`;

export const solidWGSL = `
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

@group(0) @binding(1) var<storage, read> bdata : array<vec4f>;

@group(0) @binding(2) var<storage, read> verts : array<vec4f>;

struct VsOut {
  @builtin(position) clip : vec4f,
  @location(0) normal : vec3f,
  @location(1) world  : vec3f,
  @location(2) colour : vec3f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  let p0 = verts[vi * 2u];
  let n0 = verts[vi * 2u + 1u];
  let b = u32(p0.w + 0.5) * 6u;
  let centre = bdata[b].xyz;
  let c0 = bdata[b + 2u].xyz;
  let c1 = bdata[b + 3u].xyz;
  let c2 = bdata[b + 4u].xyz;

  let world = centre + c0 * p0.x + c1 * p0.y + c2 * p0.z;
  var o : VsOut;
  o.world = world;
  o.normal = c0 * n0.x + c1 * n0.y + c2 * n0.z;
  o.colour = bdata[b + 5u].xyz;
  o.clip = V.viewProj * vec4f(world, 1.0);
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
  let spec = pow(max(dot(n, halfV), 0.0), 40.0);
  let col = in.colour * (0.25 + 0.75 * ndl) + vec3f(0.5) * spec;
  return vec4f(col, 1.0);
}
`;
