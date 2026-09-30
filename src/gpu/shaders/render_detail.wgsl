// Render pass 2 — edge-aware sharpening on display-encoded luma.
//
//   detail  = Y − Gaussian(Y, radius)
//   gate    = smoothstep(thr, 2.5·thr, |detail|)    noise must not be sharpened
//             × smoothstep(0.5·thr, 3·thr, local range)   nor smooth gradients
//   Y'      = Y + amount · mult · gate · detail
//   Y'      clamped to the 3×3 min/max ± a small margin: no overshoot, no halos
// `mult` is the per-pixel semantic × depth multiplier written by pass 1 (alpha).
// Negative: skin smoothing there instead, by −mult — an edge-preserving (bilateral) mean
// of 5 × 5 taps `s.w` px apart: blemishes and pores go, edges of features stay.

struct U { size: vec4<u32>, s: vec4<f32> } // size: W, H, HDR (1: alpha = HDR gain), _; s: amount, radius, threshold, skin smoothing's tap spacing (px)
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var gain: texture_2d<f32>; // HDR gain (r32float), or a 1×1 dummy

/** Output alpha: the HDR gain from here on (1 when HDR is off). */
fn alpha_at(p: vec2<i32>) -> f32 { return select(1.0, textureLoad(gain, p, 0).r, u.size.z != 0u); }

fn ld(p: vec2<i32>) -> vec4<f32> {
  return textureLoad(src, clamp(p, vec2<i32>(0), vec2<i32>(i32(u.size.x) - 1, i32(u.size.y) - 1)), 0);
}
fn Y(p: vec2<i32>) -> f32 { return dot(ld(p).rgb, LUMAP3); }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.size.x || id.y >= u.size.y) { return; }
  let p = vec2<i32>(id.xy);
  let c = ld(p);
  let mult = c.a;
  if (mult < -0.001) {
    let k = min(-mult, 1.0);
    let y0 = dot(c.rgb, LUMAP3);
    var acc = vec3<f32>(0.0); var ws = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
      for (var dx = -2; dx <= 2; dx++) {
        let q = ld(p + vec2<i32>(vec2<f32>(f32(dx), f32(dy)) * u.s.w)).rgb;
        let d = dot(q, LUMAP3) - y0;
        // Near in brightness counts (a feature's edge does not); farther taps count less.
        let w = exp(-d * d / (2.0 * 0.05 * 0.05)) * exp(-f32(dx * dx + dy * dy) / 8.0);
        acc += w * q; ws += w;
      }
    }
    let sm = acc / max(ws, 1e-4);
    // A little of the fine texture stays (skin, not plastic).
    textureStore(dst, p, vec4<f32>(clamp(mix(c.rgb, sm, 0.85 * k), vec3<f32>(0.0), vec3<f32>(1.0)), alpha_at(p)));
    return;
  }
  let amount = u.s.x * mult;
  if (amount <= 0.001) { textureStore(dst, p, vec4<f32>(c.rgb, alpha_at(p))); return; }
  let sigma = max(u.s.y, 0.4);
  var acc = 0.0; var wsum = 0.0;
  var mn = 1e9; var mx = -1e9;
  // Six distinct squared distances in a 5×5 window (0, 1, 2, 4, 5, 8): weights once, not 25 exp.
  let s2 = -1.0 / (2.0 * sigma * sigma);
  var wk = array<f32, 9>(1.0, exp(s2), exp(2.0 * s2), 0.0, exp(4.0 * s2), exp(5.0 * s2), 0.0, 0.0, exp(8.0 * s2));
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let y = Y(p + vec2<i32>(dx, dy));
      let w = wk[dx * dx + dy * dy];
      acc += w * y; wsum += w;
      if (abs(dx) <= 1 && abs(dy) <= 1) { mn = min(mn, y); mx = max(mx, y); }
    }
  }
  let y0 = dot(c.rgb, LUMAP3);
  let detail = y0 - acc / wsum;
  let thr = u.s.z;
  let gate = smoothstep(thr, 2.5 * thr, abs(detail)) * smoothstep(0.5 * thr, 3.0 * thr, mx - mn);
  var y1 = y0 + amount * gate * detail * 1.6;
  let margin = 0.012 * amount;
  y1 = clamp(y1, mn - margin, mx + margin);
  let outc = clamp(c.rgb + vec3<f32>(y1 - y0), vec3<f32>(0.0), vec3<f32>(1.0));
  textureStore(dst, p, vec4<f32>(outc, alpha_at(p)));
}
