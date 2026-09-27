// Burst merge (src/burst/burst.ts): full-resolution alignment refinement and a
// robust running mean of the aligned frames.
//
//   luma        frame → √(g·Y), the alignment image (noise roughly even in it)
//   align       per 16 px tile: search ±2 px around the coarse guess (from the
//               ≈1024 px scan), then one Lucas–Kanade step for the sub-pixel part
//   resample    a frame from another lens → the reference's view (colour matched;
//               what it does not cover is marked like clipped: never added)
//   init        reference → accumulator (mean = reference, weight 1)
//   weights     per pixel: how much this frame may add (0 where it disagrees with
//               the mean beyond noise — something moved — or is clipped)
//   accumulate  mean += w·(frame − mean) / (W + w)
//   finalize    accumulator → working texture; where few frames agreed (moving
//               things), mixed toward a denoised version so they are not noisier;
//               where fine texture was softened (not alignable to a fraction of a
//               pixel), the reference's own pixel kept
//
// Accumulator: per pixel two u32 = (r, g) and (b, W) as half floats. W < 0 marks
// a pixel still clipped in every frame so far.

struct U {
  w: u32, h: u32, tw: u32, th: u32,
  g: f32,        // this frame's exposure ratio to the reference
  sscale: f32,   // scan σ → σ of a 3×3 mean at full resolution
  n: f32,        // the weight a pixel gets where every frame agrees (1 = the reference alone)
  mode: u32,     // finalize: 1 = mix the denoised image into ghost areas
  sig0: vec4<f32>, sig1: vec4<f32>, // noise σ (linear) per √Y bin, 8 bins
  aff: vec4<f32>,  // another lens: reference → frame pixels, q − c_f = [a −b; b a](p − c_r) + t
  fsize: vec4<f32>, // that frame's size
  gc: vec4<f32>,   // its colour gains (rgb, another lens) and weight (a: lens detail × sharpness)
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var frame: texture_2d<f32>;
@group(0) @binding(2) var lum_out: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var ref_l: texture_2d<f32>;
@group(0) @binding(4) var frm_l: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> init_off: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> flow: array<vec4<f32>>;
@group(0) @binding(7) var samp: sampler;
@group(0) @binding(8) var<storage, read_write> acc: array<vec2<u32>>;
@group(0) @binding(9) var<storage, read_write> wbuf: array<u32>;
@group(0) @binding(10) var dn: texture_2d<f32>;
@group(0) @binding(11) var out: texture_storage_2d<rgba16float, write>;

const T: i32 = 16;      // tile
const S: i32 = 3;       // search ±2, plus 1 for the gradients of the sub-pixel step
const WIN: i32 = 22;    // T + 2·S

@compute @workgroup_size(16, 16)
fn luma(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let c = textureLoad(frame, vec2<i32>(id.xy), 0).rgb;
  textureStore(lum_out, vec2<i32>(id.xy), vec4<f32>(sqrt(max(0.0, u.g * luma2020(c))), 0.0, 0.0, 1.0));
}

var<workgroup> rt: array<f32, 256>;
var<workgroup> fw: array<f32, 484>;
var<workgroup> costs: array<f32, 49>;

fn load_l(t: texture_2d<f32>, x: i32, y: i32) -> f32 {
  return textureLoad(t, vec2<i32>(clamp(x, 0, i32(u.w) - 1), clamp(y, 0, i32(u.h) - 1)), 0).r;
}

@compute @workgroup_size(64)
fn align(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tile = wid.y * u.tw + wid.x;
  let ox = i32(wid.x) * T; let oy = i32(wid.y) * T;
  let g = init_off[tile];
  let gx = i32(round(g.x)); let gy = i32(round(g.y));
  for (var k = li; k < 256u; k += 64u) { rt[k] = load_l(ref_l, ox + i32(k % 16u), oy + i32(k / 16u)); }
  for (var k = li; k < 484u; k += 64u) { fw[k] = load_l(frm_l, ox + gx - S + i32(k % 22u), oy + gy - S + i32(k / 22u)); }
  workgroupBarrier();
  if (li < 49u) {
    let dx = i32(li % 7u); let dy = i32(li / 7u); // 0…6 = offset −3…+3
    var s = 0.0;
    for (var y = 0; y < T; y++) {
      for (var x = 0; x < T; x++) {
        let e = rt[y * T + x] - fw[(y + dy) * WIN + x + dx];
        s += e * e;
      }
    }
    costs[li] = s;
  }
  workgroupBarrier();
  if (li == 0u) {
    var bx = 3; var by = 3; var best = costs[24];
    for (var y = 1; y <= 5; y++) {
      for (var x = 1; x <= 5; x++) {
        let c = costs[y * 7 + x];
        if (c < best) { best = c; bx = x; by = y; }
      }
    }
    // Sub-pixel: one Lucas–Kanade step from the best whole-pixel offset (align.ts lkStep).
    var a = 0.0; var b = 0.0; var c = 0.0; var pp = 0.0; var qq = 0.0;
    for (var y = 0; y < T; y++) {
      for (var x = 0; x < T; x++) {
        let o = (y + by) * WIN + x + bx;
        let dgx = (fw[o + 1] - fw[o - 1]) * 0.5;
        let dgy = (fw[o + WIN] - fw[o - WIN]) * 0.5;
        let e = fw[o] - rt[y * T + x];
        a += dgx * dgx; b += dgx * dgy; c += dgy * dgy; pp += dgx * e; qq += dgy * e;
      }
    }
    let det = a * c - b * b;
    var sub = vec2<f32>(0.0);
    if (det > 1e-12) { sub = clamp(vec2<f32>(-(c * pp - b * qq), -(a * qq - b * pp)) / det, vec2<f32>(-0.5), vec2<f32>(0.5)); }
    flow[tile] = vec4<f32>(f32(gx + bx - 3) + sub.x, f32(gy + by - 3) + sub.y, best / 256.0, 0.0);
  }
}


/** Motion at a pixel: bilinear between tile centres. */
fn flow_at(p: vec2<f32>) -> vec2<f32> {
  let gx = clamp(p.x / f32(T) - 0.5, 0.0, f32(u.tw - 1u));
  let gy = clamp(p.y / f32(T) - 0.5, 0.0, f32(u.th - 1u));
  let x0 = u32(floor(gx)); let y0 = u32(floor(gy));
  let x1 = min(u.tw - 1u, x0 + 1u); let y1 = min(u.th - 1u, y0 + 1u);
  let fx = gx - f32(x0); let fy = gy - f32(y0);
  let a = mix(flow[y0 * u.tw + x0].xy, flow[y0 * u.tw + x1].xy, fx);
  let b = mix(flow[y1 * u.tw + x0].xy, flow[y1 * u.tw + x1].xy, fx);
  return mix(a, b, fy);
}

fn frame_at(p: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(frame, samp, p / vec2<f32>(f32(u.w), f32(u.h)), 0.0);
}

/**
 * The frame at p, Catmull-Rom (bicubic): a sub-pixel shift keeps the frame's detail,
 * where bilinear sampling would average neighbours (a ½-pixel shift = a 2-tap blur).
 * Never below 0 (the kernel's small overshoot).
 */
fn frame_sharp(p: vec2<f32>) -> vec3<f32> {
  let q = p - 0.5;
  let i = floor(q); let f = q - i;
  let f2 = f * f; let f3 = f2 * f;
  let w0 = -0.5 * f3 + f2 - 0.5 * f;
  let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
  let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
  let w3 = 0.5 * f3 - 0.5 * f2;
  let wx = array<f32, 4>(w0.x, w1.x, w2.x, w3.x);
  let wy = array<f32, 4>(w0.y, w1.y, w2.y, w3.y);
  let mx = vec2<i32>(i32(u.w) - 1, i32(u.h) - 1);
  var c = vec3<f32>(0.0);
  for (var y = 0; y < 4; y++) {
    var row = vec3<f32>(0.0);
    for (var x = 0; x < 4; x++) {
      let t = clamp(vec2<i32>(i) + vec2<i32>(x - 1, y - 1), vec2<i32>(0), mx);
      row += wx[x] * textureLoad(frame, t, 0).rgb;
    }
    c += wy[y] * row;
  }
  return max(c, vec3<f32>(0.0));
}

fn acc_get(x: i32, y: i32) -> vec4<f32> {
  let i = u32(clamp(y, 0, i32(u.h) - 1)) * u.w + u32(clamp(x, 0, i32(u.w) - 1));
  let v = acc[i];
  return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y));
}

fn sigma(y: f32) -> f32 {
  let b = clamp(sqrt(max(y, 0.0)) * 8.0 - 0.5, 0.0, 7.0);
  let i = u32(floor(b)); let f = b - f32(i);
  let j = min(7u, i + 1u);
  let s = array<f32, 8>(u.sig0.x, u.sig0.y, u.sig0.z, u.sig0.w, u.sig1.x, u.sig1.y, u.sig1.z, u.sig1.w);
  return mix(s[i], s[j], f);
}

fn weight_at(px: i32, py: i32) -> f32 {
  if (px >= i32(u.w)) { return 0.0; }
  let p = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);
  let f = flow_at(p);
  let x = frame_at(p + f);
  if (x.a > 0.0) { return 0.0; } // clipped in this frame
  let m = acc_get(px, py);
  if (m.w < 0.0) { return 1.0; } // clipped in every frame so far: anything unclipped is better
  // 3×3 means of the frame (aligned) and of the running mean.
  var xb = vec3<f32>(0.0); var mb = vec3<f32>(0.0);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      xb += frame_at(p + f + vec2<f32>(f32(i), f32(j))).rgb;
      mb += acc_get(px + i, py + j).rgb;
    }
  }
  xb = xb * (u.g / 9.0); mb = mb / 9.0;
  let s = sigma(luma2020(mb)) * u.sscale;
  let s2 = max(s * s * (u.g * u.g + 0.5), 1e-10);
  let d = xb - mb;
  let t = dot(d, d) / 3.0 / s2;
  // Within ~1.7σ: full weight; beyond, falling fast (a moving thing, a misalignment).
  let w = exp(-max(0.0, t - 3.0) * 0.5);
  // A frame scaled up (darker) carries more noise; a wider lens's or a shaken frame, less detail.
  return w * clamp(1.0 / (u.g * u.g), 0.25, 4.0) * u.gc.w;
}

@compute @workgroup_size(8, 8)
fn weights(@builtin(global_invocation_id) id: vec3<u32>) {
  let hw = (u.w + 1u) / 2u;
  if (id.x >= hw || id.y >= u.h) { return; }
  let x = i32(id.x * 2u); let y = i32(id.y);
  wbuf[id.y * hw + id.x] = pack2x16float(vec2<f32>(weight_at(x, y), weight_at(x + 1, y)));
}

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let hw = (u.w + 1u) / 2u;
  let ww = unpack2x16float(wbuf[id.y * hw + id.x / 2u]);
  let w = select(ww.x, ww.y, (id.x & 1u) == 1u);
  if (w < 1e-3) { return; }
  let p = vec2<f32>(vec2<u32>(id.xy)) + 0.5;
  let x = frame_sharp(p + flow_at(p)) * u.g;
  let i = id.y * u.w + id.x;
  let v = acc[i];
  var m = vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y));
  if (m.w < 0.0) { m = vec4<f32>(x, w); }
  else { let W = m.w + w; m = vec4<f32>(m.rgb + (x - m.rgb) * (w / W), W); }
  acc[i] = vec2<u32>(pack2x16float(m.rg), pack2x16float(m.ba));
}

@compute @workgroup_size(8, 8)
fn resample(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let p = vec2<f32>(vec2<u32>(id.xy)) + 0.5 - vec2<f32>(f32(u.w), f32(u.h)) * 0.5;
  let q = u.fsize.xy * 0.5 + vec2<f32>(u.aff.x * p.x - u.aff.y * p.y, u.aff.y * p.x + u.aff.x * p.y) + u.aff.zw;
  if (any(q < vec2<f32>(0.0)) || any(q > u.fsize.xy)) { textureStore(out, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0)); return; }
  // A longer lens is sampled sparsely: 3×3 samples averaged (no aliasing).
  let sc = length(u.aff.xy);
  var c = vec4<f32>(0.0);
  if (sc > 1.5) {
    let st = sc / 3.0;
    for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) { c += textureSampleLevel(frame, samp, (q + vec2<f32>(f32(i), f32(j)) * st) / u.fsize.xy, 0.0); } }
    c = c / 9.0;
  } else { c = textureSampleLevel(frame, samp, q / u.fsize.xy, 0.0); }
  textureStore(out, vec2<i32>(id.xy), vec4<f32>(c.rgb * u.gc.rgb, c.a));
}

@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let c = textureLoad(frame, vec2<i32>(id.xy), 0);
  acc[id.y * u.w + id.x] = vec2<u32>(pack2x16float(c.rg), pack2x16float(vec2<f32>(c.b, select(1.0, -1.0, c.a > 0.0))));
}

/** Noise σ (display-encoded luma, one full-size pixel) at encoded luma e: u.sig bins over 0…1 (finalize mode 1). */
fn sigma_enc(e: f32) -> f32 {
  let b = clamp(e * 8.0 - 0.5, 0.0, 7.0);
  let i = u32(floor(b)); let f = b - f32(i);
  let s = array<f32, 8>(u.sig0.x, u.sig0.y, u.sig0.z, u.sig0.w, u.sig1.x, u.sig1.y, u.sig1.z, u.sig1.w);
  return mix(s[i], s[min(7u, i + 1u)], f);
}
fn enc_y(c: vec3<f32>) -> f32 { return srgb_oetf1(clamp(u.g * luma2020(c), 0.0, 1.0)); }

@compute @workgroup_size(8, 8)
fn finalize(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let p = vec2<i32>(id.xy);
  let m = acc_get(p.x, p.y);
  var c = m.rgb;
  if (u.mode == 1u) {
    // Ghost areas (few frames agreed: something moved): toward the denoised image, as much
    // as the pixel's weight falls short of half of what full agreement gives. Where most
    // frames agreed, the merge itself is the noise reduction — no denoiser blur there.
    let want = max(0.5 * u.n, 1.0 + 1e-3);
    let a = clamp((want - abs(m.w)) / (want - 1.0), 0.0, 1.0);
    c = mix(c, textureLoad(dn, p, 0).rgb, a);
    // Detail rescue: fine texture that hand-held frames could not align to a fraction of a
    // pixel is softened by the mean. Where the merge's fine detail (pixel − 3×3 mean)
    // differs from the reference's by more than noise explains, the reference is kept.
    let mx = vec2<i32>(i32(u.w) - 1, i32(u.h) - 1);
    var rb = 0.0; var mb = 0.0;
    for (var j = -1; j <= 1; j++) {
      for (var i = -1; i <= 1; i++) {
        let q = clamp(p + vec2<i32>(i, j), vec2<i32>(0), mx);
        rb += enc_y(textureLoad(frame, q, 0).rgb);
        mb += enc_y(acc_get(q.x, q.y).rgb);
      }
    }
    let rc = textureLoad(frame, p, 0).rgb;
    let er = enc_y(rc);
    let d = abs((er - rb / 9.0) - (enc_y(m.rgb) - mb / 9.0));
    let s = max(sigma_enc(er), 1e-4);
    let keep = smoothstep(1.0 * s, 2.5 * s, d);
    c = mix(c, rc, keep);
  }
  textureStore(out, p, vec4<f32>(c, select(0.0, 1.0, m.w < 0.0)));
}
