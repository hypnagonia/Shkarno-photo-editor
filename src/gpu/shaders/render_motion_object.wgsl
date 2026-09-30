// Motion blur of a moving object (a Motion Blur layer set to Object): the object — the
// photo times its layer's mask — is smeared along its travel direction over the whole
// frame, colour and coverage alike, and laid over the background. So it streaks past its
// own outline, thinning towards the end of the streak, with the background showing
// through; inside its outline the background is the plate the engine inpainted where the
// object was (a vehicle leaves the street behind it, not a second copy of itself).
//
//   smear, pass 0   premultiplied object and coverage averaged over the travel window:
//                   t in [a, b] px along d — symmetric (trail 0) … all behind it (trail 1)
//   smear, pass 1   the same over one tap spacing of pass 0 (every pixel sampled)
//   composite       smeared object over the background (the plate inside the outline),
//                   the sharp object on top by `sharp` (a rear-curtain flash look)
//
// In linear light, as the scene motion blur (render_motion.wgsl).

struct U {
  size: vec4<u32>,  // W, H of this texture, input is linear, pass (0, 1)
  d: vec4<f32>,     // travel direction (unit, screen y down), window start a, end b (px)
  plate: vec4<f32>, // the plate's rectangle in this texture's px (x0, y0, x1, y1); empty = none
  k: vec4<f32>,     // sharp object on top 0…1, _, _, _
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var acc: texture_2d<f32>;
@group(0) @binding(4) var plate: texture_2d<f32>;
@group(0) @binding(5) var lsamp: sampler;
@group(0) @binding(6) var dst: texture_storage_2d<rgba16float, write>;

fn lin_of(c: vec3<f32>) -> vec3<f32> {
  if (u.size.z != 0u) { return max(c, vec3<f32>(0.0)); }
  return srgb_eotf(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
}
fn mask_at(p: vec2<f32>) -> f32 {
  let lim = vec2<i32>(i32(u.size.x) - 1, i32(u.size.y) - 1);
  return clamp(textureLoad(motion, clamp(vec2<i32>(floor(p)), vec2<i32>(0), lim), 0).w, 0.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn smear(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.size.x || id.y >= u.size.y) { return; }
  let px = vec2<i32>(id.xy);
  let dims = vec2<f32>(f32(u.size.x), f32(u.size.y));
  let c = vec2<f32>(px) + 0.5;
  let len = u.d.w - u.d.z;
  let n = i32(clamp(ceil(len / 2.0), 2.0, 64.0));
  var sum = vec4<f32>(0.0);
  if (u.size.w == 0u) {
    for (var k = 0; k < n; k++) {
      let p = c + u.d.xy * (u.d.z + (f32(k) + 0.5) / f32(n) * len);
      let a = mask_at(p);
      if (a > 0.0) { sum += vec4<f32>(lin_of(textureSampleLevel(src, lsamp, p / dims, 0.0).rgb) * a, a); }
    }
    sum /= f32(n);
  } else {
    // One spacing of pass 0, filled in.
    let span = len / f32(n);
    let m = i32(clamp(ceil(span), 1.0, 8.0)) + 1;
    for (var k = 0; k < m; k++) {
      let p = c + u.d.xy * (((f32(k) + 0.5) / f32(m) - 0.5) * span);
      sum += textureSampleLevel(acc, lsamp, p / dims, 0.0);
    }
    sum /= f32(m);
  }
  textureStore(dst, px, sum);
}

@compute @workgroup_size(8, 8)
fn composite(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.size.x || id.y >= u.size.y) { return; }
  let px = vec2<i32>(id.xy);
  let c0 = textureLoad(src, px, 0);
  let here = lin_of(c0.rgb);
  let a0 = mask_at(vec2<f32>(px) + 0.5);
  // The background: the photo, and inside the object's outline the plate inpainted there.
  var bg = here;
  let pr = u.plate;
  if (a0 > 0.0 && pr.z > pr.x && f32(id.x) >= pr.x && f32(id.x) < pr.z && f32(id.y) >= pr.y && f32(id.y) < pr.w) {
    let uv = (vec2<f32>(px) + 0.5 - pr.xy) / (pr.zw - pr.xy);
    bg = mix(here, srgb_eotf(clamp(textureSampleLevel(plate, lsamp, uv, 0.0).rgb, vec3<f32>(0.0), vec3<f32>(1.0))), a0);
  }
  let o = textureLoad(acc, px, 0);
  var outc = o.rgb + (1.0 - clamp(o.a, 0.0, 1.0)) * bg;
  // The object itself, sharp, on top of its own streak.
  outc = mix(outc, here, a0 * clamp(u.k.x, 0.0, 1.0));
  if (u.size.z == 0u) { outc = srgb_oetf(clamp(outc, vec3<f32>(0.0), vec3<f32>(1.0))); }
  textureStore(dst, px, vec4<f32>(outc, c0.a));
}
