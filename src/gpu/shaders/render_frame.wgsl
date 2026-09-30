// The frame (src/geometry/frame.ts): the finished preview turned, mirrored, straightened
// and cropped — each output pixel read from the rendered photo through one affine map,
// bilinear. Outside the photo (straightened corners, while the crop is being set): dark.

struct U {
  size: vec4<u32>, // output W, H; the photo's rows start at `top` of the source texture, its W, H
  ax: vec4<f32>,   // x = ax.x·u + ax.y·v + ax.z (0…1 of the photo)
  ay: vec4<f32>,   // y = ay.x·u + ay.y·v + ay.z
  src: vec4<f32>,  // photo W, H (px), top row, _
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var srcT: texture_2d<f32>;
@group(0) @binding(2) var lsamp: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.size.x || id.y >= u.size.y) { return; }
  let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(u.size.x), f32(u.size.y));
  let s = vec2<f32>(dot(u.ax.xy, uv) + u.ax.z, dot(u.ay.xy, uv) + u.ay.z);
  var c = vec4<f32>(0.09, 0.09, 0.1, 1.0);
  if (all(s >= vec2<f32>(0.0)) && all(s <= vec2<f32>(1.0))) {
    let dims = vec2<f32>(textureDimensions(srcT));
    let px = s * u.src.xy + vec2<f32>(0.0, u.src.z);
    c = vec4<f32>(textureSampleLevel(srcT, lsamp, px / dims, 0.0).rgb, 1.0);
  }
  textureStore(dst, vec2<i32>(id.xy), c);
}
