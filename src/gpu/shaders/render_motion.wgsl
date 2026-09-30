// Motion blur (Blur layers set to Motion): each pixel averages the image along its
// layer's direction (into the depth: away from the vanishing point) over a streak of
// its layer's length, in linear light (a bright light streaks as light, not as grey),
// evenly — an exposure sees every point of the path for as long as any other, so the
// streaks have crisp ends. Two passes: the second averages over one spacing of the
// first's taps, so a long streak is sampled at every pixel (no stepped ghosts) at the
// cost of 64 + 8 taps. Taps are weighted by how much they themselves are blurred: a
// sharp subject does not smear into the streaked background around it, as when the
// camera pans with it.

struct U {
  size: vec4<u32>, // W, H of this texture, input is linear, _
  m: vec4<f32>,    // streak length at amount 1 (output px), vanishing point (px of this texture), pass (0, 1)
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
  let mv3 = textureLoad(motion, px, 0).xyz;
  let mv = mv3.xy;
  // Through the mask: the mask's weight here (1 otherwise); every tap counts the same then.
  let through = mv3.z < 0.999;
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
  let n = i32(clamp(ceil(len / 2.0), 4.0, 64.0));
  // Pass 0: the whole streak in n taps; pass 1: one tap spacing of pass 0, filled in.
  let span = select(len, len / f32(n), u.m.w > 0.5);
  let taps = select(n, i32(clamp(ceil(len / f32(n)), 1.0, 8.0)) + 1, u.m.w > 0.5);
  var acc = vec3<f32>(0.0); var ws = 0.0;
  for (var k = 0; k < taps; k++) {
    let t = (f32(k) + 0.5) / f32(taps) - 0.5; // −½ … ½ of the span, evenly
    let p = vec2<f32>(px) + 0.5 + d * (t * span);
    let q = clamp(vec2<i32>(floor(p)), vec2<i32>(0), lim);
    let w = select(mix(0.05, 1.0, clamp(textureLoad(motion, q, 0).x / max(mv.x, 1e-3), 0.0, 1.0)), 1.0, through);
    acc += w * lin_of(textureSampleLevel(src, lsamp, p / dims, 0.0).rgb);
    ws += w;
  }
  var outc = mix(lin_of(c0.rgb), acc / ws, clamp(mv3.z, 0.0, 1.0));
  if (u.size.z == 0u) { outc = srgb_oetf(clamp(outc, vec3<f32>(0.0), vec3<f32>(1.0))); }
  textureStore(dst, px, vec4<f32>(outc, c0.a));
}
