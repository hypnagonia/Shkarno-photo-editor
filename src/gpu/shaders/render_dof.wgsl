// Optional depth-aware depth of field (gather, linear light).
//
// Circle of confusion grows continuously with |distance − focus| — separately
// shaped in front of and behind the focal plane — never a binary mask.
// Background samples cannot spread over nearer, sharper pixels (depth test),
// which is what keeps subject edges clean. Large radii read from a mip chain
// so 48 taps stay smooth.

struct U {
  size: vec4<u32>,     // W, H, guide w, guide h
  d: vec4<f32>,        // focus distance, max radius px, near scale, automatic subject's depth extent below the focus
  f: vec4<f32>,        // focus count, zone mode, mip count, automatic subject's depth extent above the focus
  foci: array<vec4<f32>, 2>, // up to 8 focus points: near end of each object's depth range
  zc: array<vec4<f32>, 2>,   // 4 inner depth-zone boundaries (zc[0]) — zone mode when f.y = 1
  zv: array<vec4<f32>, 2>,   // blur 0..1 per zone
  fociHi: array<vec4<f32>, 2>, // … and the far end (a point on a flat spot: both the same)
  b: vec4<f32>,        // Blur layers' radius px at amount 1, depth of field on (1) / off (0), bokeh 0…1, aperture blades (0 round)
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var src: texture_2d<f32>;       // mip chain of the sharpened image (linear P3)
@group(0) @binding(2) var distt: texture_2d<f32>;     // per-pixel refined distance (r32float)
@group(0) @binding(3) var coct: texture_2d<f32>;      // per-pixel (distance, circle of confusion): written by `coc_pass`
@group(0) @binding(6) var cocdst: texture_storage_2d<rg32float, write>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var dst: texture_storage_2d<rgba16float, write>;

fn dist_at(px: vec2<f32>) -> f32 {
  let m = vec2<i32>(i32(u.size.x) - 1, i32(u.size.y) - 1);
  return textureLoad(distt, clamp(vec2<i32>(px), vec2<i32>(0), m), 0).r;
}

fn coc1(d: f32, lo: f32, hi: f32) -> f32 {
  // Everything within the object's own depth range [lo, hi] is sharp; blur grows
  // behind its far end and in front of its near end. The ramp spans the depth
  // range left behind (distance is linear in disparity, so a near foreground
  // squeezes a subject and its room toward 1).
  let behind = smoothstep(0.0, clamp((1.0 - hi) * 0.9, 0.12, 0.55), d - hi);
  let front = u.d.z * smoothstep(0.0, 0.4, lo - d);
  return max(behind, front);
}

// Several focus points: a pixel is as sharp as its nearest focal distance allows.
/** Manual depth zones: one blur value per zone, with a short smooth transition
 * (±0.02 in distance) only at each boundary — a layer is blurred uniformly. */
fn zone_blur(d: f32) -> f32 {
  var v = u.zv[0].x;
  for (var i = 0u; i < 4u; i++) {
    let e = u.zc[0][i];
    let next = u.zv[(i + 1u) / 4u][(i + 1u) % 4u];
    v = mix(v, next, smoothstep(e - 0.02, e + 0.02, d));
  }
  return v;
}

fn coc(d: f32) -> f32 {
  if (u.f.y > 0.5) { return u.d.y * zone_blur(d); }
  let n = u32(u.f.x);
  if (n == 0u) { return u.d.y * coc1(d, u.d.x - u.d.w, u.d.x + u.f.w); }
  var c = 1.0;
  for (var i = 0u; i < n; i++) {
    c = min(c, coc1(d, u.foci[i / 4u][i % 4u], u.fociHi[i / 4u][i % 4u]));
  }
  return u.d.y * c;
}

/** Distance and blur radius of every pixel, once (the gather reads 48 neighbours' each). */
@compute @workgroup_size(8, 8)
fn coc_pass(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.size.x || id.y >= u.size.y) { return; }
  // Distance in 0…1, Blur layers' amount packed above it (render_tone.wgsl).
  let v = textureLoad(distt, vec2<i32>(id.xy), 0).r;
  let k = floor(v / 2.0);
  let d = v - 2.0 * k;
  let depth_coc = select(0.0, coc(d), u.b.y > 0.5);
  textureStore(cocdst, vec2<i32>(id.xy), vec4<f32>(d, max(depth_coc, (k / 1000.0) * u.b.x), 0.0, 0.0));
}

/**
 * Bokeh: how bright a sample is as a light — 0 … 1 from about a mid grey up (linear
 * brightness, HDR excess included). A light counts up to ×30 in the gather (it fills its
 * disc instead of fading into what surrounds it) and shines up to ×2.5 brighter (a lamp is
 * far brighter than the picture can show: its disc stays bright when spread wide).
 */
fn light_of(s: vec4<f32>) -> f32 {
  if (u.b.z <= 0.0) { return 0.0; }
  let y = dot(max(s.rgb, vec3<f32>(0.0)), LUMAP3) + max(s.a, 0.0);
  return u.b.z * smoothstep(0.06, 0.45, y);
}
/** A polygonal aperture: the disc's edge distance at angle `a` (1 at the corners), round when no blades. */
fn aperture(a: f32) -> f32 {
  let n = u.b.w;
  if (n < 3.0) { return 1.0; }
  let seg = 6.2831853 / n;
  let t = a - seg * floor(a / seg) - seg * 0.5;
  return cos(seg * 0.5) / cos(t);
}

fn coc_at(px: vec2<f32>) -> vec2<f32> {
  let m = vec2<i32>(i32(u.size.x) - 1, i32(u.size.y) - 1);
  return textureLoad(coct, clamp(vec2<i32>(px), vec2<i32>(0), m), 0).rg;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = u.size.x; let H = u.size.y;
  if (id.x >= W || id.y >= H) { return; }
  let px = vec2<f32>(id.xy) + 0.5;
  let cc = coc_at(px);
  let dc = cc.x;
  let rc = cc.y;
  let center = textureLoad(src, vec2<i32>(id.xy), 0);
  // Alpha arrives as HDR excess luminance (output.wgsl linearize) and leaves as a gain again.
  if (rc < 0.6) { textureStore(dst, vec2<i32>(id.xy), vec4<f32>(center.rgb, 1.0 + center.a / max(dot(center.rgb, LUMAP3), 1e-6))); return; }
  let size = vec2<f32>(f32(W), f32(H));
  // (The centre as one ordinary tap: a lamp's own sharp core must not show through its disc.)
  var acc = center * (1.0 + 1.5 * light_of(center)); var ws = 1.0;
  // Bokeh: more taps, the spiral turned at random per pixel (a highlight's disc is filled
  // evenly, not printed with the spiral's pattern).
  let N = select(48, 128, u.b.z > 0.0);
  let golden = 2.39996323;
  let spin = select(0.0, fract(sin(dot(px, vec2<f32>(12.9898, 78.233))) * 43758.547) * 6.2831853, u.b.z > 0.0);
  // (With bokeh a level finer, the taps being denser: the discs keep their edge.)
  let level = clamp(log2(max(rc, 1.0) / 5.0) - select(0.0, 1.0, u.b.z > 0.0), 0.0, u.f.z - 1.0);
  for (var i = 1; i <= N; i++) {
    let r = rc * sqrt(f32(i) / f32(N));
    let a = f32(i) * golden + spin;
    let o = vec2<f32>(cos(a), sin(a)) * r;
    let sp = px + o;
    if (sp.x < 0.0 || sp.y < 0.0 || sp.x >= size.x || sp.y >= size.y) { continue; }
    let cs = coc_at(sp);
    let ds = cs.x;
    let rs = cs.y * aperture(a + 0.5236);
    // A sample contributes if its own blur disc (its aperture's shape) reaches the centre…
    var w = smoothstep(r - 1.0, r + 1.0, rs);
    // …and background behind a sharper centre may not spill onto it.
    if (ds > dc + 0.04) { w *= smoothstep(r - 1.0, r + 1.0, rc); w = min(w, smoothstep(0.0, 0.1, rc / max(u.d.y, 1.0))); }
    let s = textureSampleLevel(src, samp, sp / size, level);
    let ls = light_of(s);
    w *= 1.0 + 29.0 * ls;
    acc += s * (1.0 + 1.5 * ls) * w; ws += w;
  }
  let m = acc / ws;
  textureStore(dst, vec2<i32>(id.xy), vec4<f32>(m.rgb, 1.0 + m.a / max(dot(m.rgb, LUMAP3), 1e-6)));
}
