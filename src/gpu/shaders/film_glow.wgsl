// Film glow at low resolution (see render_film.wgsl): the light above display white
// from the whole frame (glow_src), then a separable Gaussian with two radii (glow_blur):
// bloom in rgb, halation in a.

struct GU {
  size: vec4<u32>,  // this pass's W, H; source W, H (glow_src) / horizontal (glow_blur)
  wb0: vec4<f32>, wb1: vec4<f32>, wb2: vec4<f32>, // white balance rows (glow_src)
  k: vec4<f32>,     // exposure gain, bloom threshold, halation threshold, a clipped light's excess (glow_src)
  s: vec4<f32>,     // σ bloom (rgb), σ halation (a), in texels (glow_blur)
}
@group(0) @binding(0) var<uniform> gu: GU;
@group(0) @binding(1) var gsrc: texture_2d<f32>;
@group(0) @binding(2) var gdst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn glow_src(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= gu.size.x || id.y >= gu.size.y) { return; }
  // A 4×4 average of the source over this texel's footprint.
  let sc = vec2<f32>(f32(gu.size.z), f32(gu.size.w)) / vec2<f32>(f32(gu.size.x), f32(gu.size.y));
  let lim = vec2<i32>(i32(gu.size.z) - 1, i32(gu.size.w) - 1);
  var acc = vec4<f32>(0.0);
  for (var j = 0; j < 4; j++) {
    for (var i = 0; i < 4; i++) {
      let p = vec2<i32>((vec2<f32>(id.xy) + (vec2<f32>(f32(i), f32(j)) + 0.5) / 4.0) * sc);
      acc += max(textureLoad(gsrc, min(p, lim), 0), vec4<f32>(0.0));
    }
  }
  let c0 = acc.rgb / 16.0;
  let clipped = acc.a / 16.0; // share of the footprint the sensor clipped: a light source
  let c = vec3<f32>(dot(gu.wb0.xyz, c0), dot(gu.wb1.xyz, c0), dot(gu.wb2.xyz, c0)) * gu.k.x;
  let Y = max(dot(c, vec3<f32>(0.2627, 0.678, 0.0593)), 1e-6);
  // Only the light beyond a threshold spreads (soft knee), in the light's own colour. A
  // clipped light was brighter than the sensor recorded — by how much is unknown: a few
  // stops, as street lamps and the sun's reflections are.
  // (In units of the threshold, so a dim night frame's lamps glow like a day frame's sun.)
  let b = max(Y / gu.k.y - 1.0, 0.0);
  let h = max(Y / gu.k.z - 1.0, 0.0);
  // Logarithmic: a lamp a hundred times over the threshold glows more than one just over it,
  // not a hundred times more (the eye, and the print, see it in stops).
  let bl = log2(1.0 + b * b / (b + 0.5)) + clipped * gu.k.w * 0.5;
  let hl = log2(1.0 + h * h / (h + 0.5)) + clipped * gu.k.w;
  textureStore(gdst, vec2<i32>(id.xy), vec4<f32>(c / Y * bl, hl));
}

@compute @workgroup_size(8, 8)
fn glow_blur(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= gu.size.x || id.y >= gu.size.y) { return; }
  let sb = max(gu.s.x, 0.5); let sh = max(gu.s.y, 0.5);
  let r = i32(min(ceil(max(sb, sh) * 3.0), 48.0));
  let d = select(vec2<i32>(0, 1), vec2<i32>(1, 0), gu.size.z == 1u);
  let lim = vec2<i32>(i32(gu.size.x) - 1, i32(gu.size.y) - 1);
  var ab = vec3<f32>(0.0); var ah = 0.0; var wb = 0.0; var wh = 0.0;
  for (var t = -r; t <= r; t++) {
    let v = textureLoad(gsrc, clamp(vec2<i32>(id.xy) + d * t, vec2<i32>(0), lim), 0);
    let x = f32(t * t);
    let w1 = exp(-x / (2.0 * sb * sb)); let w2 = exp(-x / (2.0 * sh * sh));
    ab += w1 * v.rgb; wb += w1; ah += w2 * v.a; wh += w2;
  }
  textureStore(gdst, vec2<i32>(id.xy), vec4<f32>(ab / wb, ah / wh));
}
