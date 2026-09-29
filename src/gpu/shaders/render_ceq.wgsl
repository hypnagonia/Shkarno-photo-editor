// Contrast equalizer (src/tone/contrastEq.ts): an edge-avoiding à-trous wavelet
// transform of the tone pass's output, in OkLab, with a gain per detail band.
//
//   level     s_{j+1} = B-spline 5×5 of s_j at spacing `step` (render px, may be
//             fractional: sampled bilinearly), weights falling across lightness steps;
//             acc += gain_j · (s_j − s_{j+1})  (lightness gain on L, colour gain on a, b)
//             The first level reads the tone output (display-encoded P3) and starts acc.
//   finalize  acc → display-encoded P3 (alpha: the tone pass's sharpening multiplier, kept)
//
// acc is a storage buffer of packed halves (read and written in place).

struct U {
  size: vec4<u32>,   // W, H (this target), first level (1/0), _
  lvl: vec4<f32>,    // step (render px), lightness gain, colour gain, edge σ (OkLab L)
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<storage, read_write> acc: array<vec2<u32>>;

const P3_FROM_SRGB_C = mat3x3<f32>(
  vec3<f32>(0.8224621, 0.0331941, 0.0170827),
  vec3<f32>(0.1775380, 0.9668058, 0.0723974),
  vec3<f32>(0.0, 0.0, 0.9105199));

fn enc_to_lab_c(e: vec3<f32>) -> vec3<f32> { return lin_srgb_to_oklab(P3_TO_SRGB * srgb_eotf(clamp(e, vec3<f32>(0.0), vec3<f32>(1.0)))); }
fn lab_to_enc_c(l: vec3<f32>) -> vec3<f32> {
  var p3 = P3_FROM_SRGB_C * oklab_to_lin_srgb(l);
  // Out of gamut: toward the pixel's own luminance, never clipped per channel.
  let Y = dot(p3, LUMAP3);
  let mn = min(p3.r, min(p3.g, p3.b));
  if (mn < 0.0) { p3 = mix(p3, vec3<f32>(max(Y, 0.0)), clamp(-mn / max(Y - mn, 1e-6), 0.0, 1.0)); }
  return srgb_oetf(clamp(p3, vec3<f32>(0.0), vec3<f32>(1.0)));
}

/** s_j at a (fractional) render position: OkLab + alpha. */
fn s_at(p: vec2<f32>) -> vec4<f32> {
  let c = textureSampleLevel(src, samp, p / vec2<f32>(f32(u.size.x), f32(u.size.y)), 0.0);
  if (u.size.z == 1u) { return vec4<f32>(enc_to_lab_c(c.rgb), c.a); }
  return c;
}

@compute @workgroup_size(8, 8)
fn level(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.size.x || id.y >= u.size.y) { return; }
  let p = vec2<f32>(id.xy) + 0.5;
  let c = s_at(p);
  let k = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
  let inv2s2 = 1.0 / (2.0 * u.lvl.w * u.lvl.w);
  var sum = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var j = -2; j <= 2; j++) {
    for (var i = -2; i <= 2; i++) {
      let q = s_at(p + vec2<f32>(f32(i), f32(j)) * u.lvl.x);
      let dL = q.x - c.x;
      let w = k[i + 2] * k[j + 2] * exp(-dL * dL * inv2s2);
      sum += w * q.xyz;
      wsum += w;
    }
  }
  let s1 = sum / max(wsum, 1e-6);
  textureStore(dst, vec2<i32>(id.xy), vec4<f32>(s1, c.a));
  // The band this level separates, scaled: lightness by one gain, colour by the other.
  let d = c.xyz - s1;
  let i = id.y * u.size.x + id.x;
  var a = vec4<f32>(c.xyz, c.a);
  if (u.size.z == 0u) { let v = acc[i]; a = vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y)); }
  a = vec4<f32>(a.x + u.lvl.y * d.x, a.yz + u.lvl.z * d.yz, a.w);
  acc[i] = vec2<u32>(pack2x16float(a.xy), pack2x16float(a.zw));
}

@compute @workgroup_size(8, 8)
fn finalize(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.size.x || id.y >= u.size.y) { return; }
  let v = acc[id.y * u.size.x + id.x];
  let a = vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y));
  textureStore(dst, vec2<i32>(id.xy), vec4<f32>(lab_to_enc_c(vec3<f32>(clamp(a.x, 0.0, 1.0), a.yz)), a.w));
}
