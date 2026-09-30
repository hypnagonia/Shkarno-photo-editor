// Motion blur (Blur layers set to Motion): each pixel averages the image along its
// layer's direction (by depth: away from the vanishing point) over a streak of its
// layer's length, in linear light (a bright
// light streaks as light, not as grey). Taps are weighted by how much they themselves
// are blurred: a sharp subject does not smear into the streaked background around it,
// as when the camera pans with it.

struct U {
  size: vec4<u32>, // W, H of this texture, input is linear, _
  m: vec4<f32>,    // streak length at amount 1 (output px), vanishing point (px of this texture)
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var lsamp: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;

fn lin_of(c: vec3<f32>) -> vec3<f32> {
  if (u.size.z != 0u) { return max(c, vec3<f32>(0.0)); }
  return srgb_eotf(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.size.x || id.y >= u.size.y) { return; }
  let px = vec2<i32>(id.xy);
  let c0 = textureLoad(src, px, 0);
  let mv = textureLoad(motion, px, 0).xy;
  let len = mv.x * u.m.x;
  if (len < 0.75) { textureStore(dst, px, c0); return; }
  let dims = vec2<f32>(f32(u.size.x), f32(u.size.y));
  let lim = vec2<i32>(i32(u.size.x) - 1, i32(u.size.y) - 1);
  // (Screen y points down: a positive angle streaks up and to the right, as on a compass.)
  var d = vec2<f32>(cos(mv.y), -sin(mv.y));
  // Angle 100: by depth — along the ray from the vanishing point (perspective).
  if (mv.y > 50.0) {
    let r = vec2<f32>(px) + 0.5 - u.m.yz;
    d = select(vec2<f32>(1.0, 0.0), r / max(length(r), 1e-3), length(r) > 1.0);
  }
  let n = i32(clamp(ceil(len / 1.5), 4.0, 48.0));
  var acc = vec3<f32>(0.0); var ws = 0.0;
  for (var k = 0; k <= n; k++) {
    let t = f32(k) / f32(n) - 0.5; // −½ … ½ of the streak
    let p = vec2<f32>(px) + 0.5 + d * (t * len);
    // Slightly tapered ends: a box streak has hard-edged ghosts.
    var w = 1.0 - 0.5 * abs(t * 2.0);
    let q = clamp(vec2<i32>(floor(p)), vec2<i32>(0), lim);
    w *= mix(0.05, 1.0, clamp(textureLoad(motion, q, 0).x / max(mv.x, 1e-3), 0.0, 1.0));
    acc += w * lin_of(textureSampleLevel(src, lsamp, p / dims, 0.0).rgb);
    ws += w;
  }
  var outc = acc / ws;
  if (u.size.z == 0u) { outc = srgb_oetf(clamp(outc, vec3<f32>(0.0), vec3<f32>(1.0))); }
  textureStore(dst, px, vec4<f32>(outc, c0.a));
}
