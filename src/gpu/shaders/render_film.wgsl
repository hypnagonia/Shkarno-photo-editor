// Film — the last pass before output (src/film/film.ts chooses the numbers).
//
//   glow_src   the scene's light above display white, from the whole frame at low
//              resolution (so an export in strips sees every light, not only its
//              strip's): rgb = what a diffusion filter spreads (bloom), a = what
//              reaches the base of the film and comes back red (halation)
//   glow_blur  separable Gaussian, two radii at once: bloom (rgb) is optical and wide,
//              halation (a) is tight and scales with the negative (a fixed distance in
//              the emulsion is a larger share of a small frame)
//   main       on the finished image: softness (the film's MTF, a Gaussian of a size
//              fixed on the film), the glow added as light, a gentler shoulder in the
//              highlights, then grain
//
// Grain follows a particle model of the emulsion (after spektrafilm's, our own code):
// each dye layer's density D fluctuates with the number of developed grains, so its
// RMS grows as √D (Selwyn), in dye clouds of a size fixed on the film; the three
// layers are partly independent (colour grain), the blue-sensitive one coarser, and
// clumps modulate them (micro-structure). The density noise reaches the print through
// the negative's gamma: Δlog10(exposure) = ΔD / γ. Sizes are in full-image pixels, so
// strips, the preview and the export share one pattern; an output pixel covering F
// full-image pixels averages the grain it covers (what shrinking the export shows).

struct U {
  size: vec4<u32>,  // W, H of this texture, row offset in the full image, input is linear
  g: vec4<f32>,     // grain: σ_D per full-image pixel, dye cloud (full-image px), clumping, layer correlation
  f: vec4<f32>,     // footprint (full-image px per output px), draft, softness σ (output px), shoulder
  h: vec4<f32>,     // halation amount, bloom amount, glow on, image height (output px)
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var glow: texture_2d<f32>;
@group(0) @binding(4) var lsamp: sampler;

const TAU = 6.2831853;
const LN10 = 2.3025851;
const GAMMA_NEG = 0.65;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn grad(i: vec2<i32>, seed: u32) -> vec2<f32> {
  let h = pcg(pcg(bitcast<u32>(i.x) ^ seed) + bitcast<u32>(i.y) * 0x9e3779b9u);
  let a = f32(h) * (TAU / 4294967296.0);
  return vec2<f32>(cos(a), sin(a));
}
/** Gradient noise, unit RMS. */
fn gnoise(x: vec2<f32>, seed: u32) -> f32 {
  let i = vec2<i32>(floor(x));
  let f = x - floor(x);
  let q = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let n00 = dot(grad(i, seed), f);
  let n10 = dot(grad(i + vec2<i32>(1, 0), seed), f - vec2<f32>(1.0, 0.0));
  let n01 = dot(grad(i + vec2<i32>(0, 1), seed), f - vec2<f32>(0.0, 1.0));
  let n11 = dot(grad(i + vec2<i32>(1, 1), seed), f - vec2<f32>(1.0, 1.0));
  return mix(mix(n00, n10, q.x), mix(n01, n11, q.x), q.y) * 4.63;
}

fn load_lin(p: vec2<i32>) -> vec3<f32> {
  let q = clamp(p, vec2<i32>(0), vec2<i32>(i32(u.size.x) - 1, i32(u.size.y) - 1));
  let c = textureLoad(src, q, 0).rgb;
  if (u.size.w != 0u) { return max(c, vec3<f32>(0.0)); }
  return srgb_eotf(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.size.x || id.y >= u.size.y) { return; }
  let px = vec2<i32>(id.xy);
  let alpha = textureLoad(src, px, 0).a;
  var lin = load_lin(px);

  // Softness: the film's own blur, a Gaussian of σ (output pixels), up to 7×7.
  let sg = u.f.z;
  if (sg > 0.25 && u.f.y < 0.5) {
    let r = i32(min(ceil(sg * 2.5), 3.0));
    var acc = vec3<f32>(0.0); var ws = 0.0;
    for (var dy = -r; dy <= r; dy++) {
      for (var dx = -r; dx <= r; dx++) {
        let w = exp(-f32(dx * dx + dy * dy) / (2.0 * sg * sg));
        acc += w * load_lin(px + vec2<i32>(dx, dy)); ws += w;
      }
    }
    lin = acc / ws;
  }

  // Glow, as light: bloom keeps the light's colour; halation comes back through the red layer.
  if (u.h.z > 0.5) {
    let uv = (vec2<f32>(f32(id.x), f32(id.y + u.size.z)) + 0.5) / vec2<f32>(f32(u.size.x), u.h.w);
    let gl = textureSampleLevel(glow, lsamp, uv, 0.0);
    lin += u.h.y * (REC2020_TO_P3 * gl.rgb) + u.h.x * gl.a * vec3<f32>(1.0, 0.3, 0.08);
  }

  var e = srgb_oetf(clamp(lin, vec3<f32>(0.0), vec3<f32>(1.0)));
  // A gentler shoulder (print paper): the upper highlights come down a little, white stays white.
  if (u.f.w > 0.0) {
    let t = clamp((e - vec3<f32>(0.6)) / 0.4, vec3<f32>(0.0), vec3<f32>(1.0));
    e -= u.f.w * 0.06 * sin(t * 3.14159265);
  }

  // Grain.
  if (u.g.x > 0.0) {
    let F = max(u.f.x, 1.0);
    let o = vec2<f32>(f32(id.x), f32(id.y + u.size.z)) * F;
    let k = min(select(4u, 2u, u.f.y > 0.5), u32(ceil(F - 1e-3)));
    let s = max(u.g.y, 0.7);
    var nl = 0.0; var nc = vec3<f32>(0.0); var cl = 0.0;
    for (var j = 0u; j < k; j++) {
      for (var i = 0u; i < k; i++) {
        let p = floor(o + (vec2<f32>(f32(i), f32(j)) + 0.5) * (F / f32(k))) + 0.5;
        nl += gnoise(p / s, 0x2545f491u);
        // Per layer: red, green, blue-sensitive (the fast blue layer has the coarsest grain).
        nc += vec3<f32>(gnoise(p / s, 0x9e3779b1u), gnoise(p / (s * 0.85), 0x85ebca77u), gnoise(p / (s * 1.4), 0xc2b2ae3du));
        cl += gnoise(p / (s * 3.5), 0x51ed27u);
      }
    }
    let norm = min(1.0, f32(k) / F) / f32(k * k);
    nl *= norm; nc *= norm;
    cl /= f32(k * k); // clumps are coarse: their field is not thinned by the footprint
    let rho = clamp(u.g.w, 0.0, 1.0);
    let n = rho * vec3<f32>(nl) + sqrt(1.0 - rho * rho) * nc;
    let clump = exp(u.g.z * 0.5 * cl);
    let linE = max(srgb_eotf(clamp(e, vec3<f32>(0.0), vec3<f32>(1.0))), vec3<f32>(1e-4));
    let D = clamp(0.15 + GAMMA_NEG * (log10v(linE) + 2.4), vec3<f32>(0.1), vec3<f32>(2.2));
    let dD = u.g.x * sqrt(D) * clump * n;
    let fade = smoothstep(vec3<f32>(0.0), vec3<f32>(0.04), e) * (vec3<f32>(1.0) - smoothstep(vec3<f32>(0.96), vec3<f32>(1.0), e));
    // Through the print: its slope, steepest in the mid-tones (paper toe and shoulder), turns
    // Δlog exposure into a visible step — so the grain is plainest in the mid-tones.
    e += 2.0 * e * (vec3<f32>(1.0) - e) * (LN10 / 2.2) * dD / GAMMA_NEG * fade;
  }
  e = clamp(e, vec3<f32>(0.0), vec3<f32>(1.0));
  var outc = e;
  if (u.size.w != 0u) { outc = srgb_eotf(e); }
  textureStore(dst, px, vec4<f32>(outc, alpha));
}

fn log10v(x: vec3<f32>) -> vec3<f32> { return log2(x) * 0.30103; }
