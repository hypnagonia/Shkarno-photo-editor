// Light rays (Light Rays layers): the light the layers' masks let out — the bright parts
// of the picture there, above a threshold — streaked away from the light source, each
// pixel gathering what lies between it and the source (a radial blur toward the source,
// fading along the way), then added on top. In linear light.
//
//   pass 0   the gather: n taps from the pixel toward the source over `len` of the way
//   pass 1   the same over one tap spacing of pass 0 (every pixel sampled: no banding)
//   add      the picture plus the rays, by strength

struct U {
  size: vec4<u32>, // W, H of this texture, input is linear, pass (0, 1, 2 = add)
  s: vec4<f32>,    // the source (px of this texture), share of the way the rays reach (0…1), strength
  k: vec4<f32>,    // threshold (display brightness 0…1), _, _, _
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var acc: texture_2d<f32>;
@group(0) @binding(4) var lsamp: sampler;
@group(0) @binding(5) var dst: texture_storage_2d<rgba16float, write>;

fn lin_of(c: vec3<f32>) -> vec3<f32> {
  if (u.size.z != 0u) { return max(c, vec3<f32>(0.0)); }
  return srgb_eotf(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
}
/** The light let out at p: the mask (w's high bits, render_tone.wgsl) times what is above the threshold. */
fn light_at(p: vec2<f32>, dims: vec2<f32>) -> vec3<f32> {
  let lim = vec2<i32>(i32(u.size.x) - 1, i32(u.size.y) - 1);
  let q = clamp(vec2<i32>(floor(p)), vec2<i32>(0), lim);
  let m = floor(textureLoad(motion, q, 0).w / 32.0) / 31.0;
  if (m <= 0.0) { return vec3<f32>(0.0); }
  let c = lin_of(textureSampleLevel(src, lsamp, p / dims, 0.0).rgb);
  // Brightness as displayed (the threshold is judged by eye), how far above the threshold.
  let y = pow(clamp(dot(c, vec3<f32>(0.2290, 0.6917, 0.0793)), 0.0, 1.0), 1.0 / 2.2);
  let a = smoothstep(u.k.x, min(1.0, u.k.x + 0.25), y);
  return c * a * m;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.size.x || id.y >= u.size.y) { return; }
  let px = vec2<i32>(id.xy);
  let dims = vec2<f32>(f32(u.size.x), f32(u.size.y));
  let c = vec2<f32>(px) + 0.5;
  let r = c - u.s.xy;
  let dist = length(r);
  if (u.size.w == 2u) {
    let c0 = textureLoad(src, px, 0);
    var o = lin_of(c0.rgb) + u.s.w * textureLoad(acc, px, 0).rgb;
    if (u.size.z == 0u) { o = srgb_oetf(clamp(o, vec3<f32>(0.0), vec3<f32>(1.0))); }
    textureStore(dst, px, vec4<f32>(o, c0.a));
    return;
  }
  let lpx = u.s.z * dist;
  let n = i32(clamp(ceil(lpx / 2.0), 2.0, 64.0));
  var sum = vec3<f32>(0.0);
  if (u.size.w == 0u) {
    // Toward the source; nearer the pixel counts most (the ray fades as it travels).
    var ws = 0.0;
    for (var i = 0; i < n; i++) {
      let t = (f32(i) + 0.5) / f32(n);
      let wt = 1.0 - t;
      sum += wt * light_at(u.s.xy + r * (1.0 - t * u.s.z), dims);
      ws += wt;
    }
    sum /= max(ws, 1e-4);
  } else {
    let span = lpx / f32(n);
    let m = i32(clamp(ceil(span), 1.0, 8.0)) + 1;
    let dir = r / max(dist, 1e-3);
    for (var i = 0; i < m; i++) {
      let p = c + dir * (((f32(i) + 0.5) / f32(m) - 0.5) * span);
      sum += textureSampleLevel(acc, lsamp, p / dims, 0.0).rgb;
    }
    sum /= f32(m);
  }
  textureStore(dst, px, vec4<f32>(sum, 1.0));
}
