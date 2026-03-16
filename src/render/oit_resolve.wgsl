// Fullscreen resolve for weighted blended OIT: resolve accum then composite over scene.
// accum.rgb = sum(color_i * alpha_i * w_i), accum.a = sum(alpha_i * w_i).
// oit_rgb = accum.rgb / accum.a (weighted average color).
// oit_alpha uses accum_alpha_scale, resolve_opacity_power, opacity_bias for tuning.
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var accum_tex: texture_2d<f32>;
@group(0) @binding(2) var resolve_sampler: sampler;
// Layout must match OitSettingsUniform (only resolve fields used here).
struct OitResolveParams {
    _depth_weight_scale: f32,
    _min_weight: f32,
    _depth_weight_power: f32,
    _opacity_scale: f32,
    accum_alpha_scale: f32,
    resolve_opacity_power: f32,
    opacity_bias: f32,
    _pad0: f32,
    _half_res: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
@group(0) @binding(3) var<uniform> oit_resolve: OitResolveParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(2.0 * x - 1.0, 2.0 * y - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_resolve(input: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_tex, resolve_sampler, input.uv);
    let accum = textureSample(accum_tex, resolve_sampler, input.uv);
    let weight = max(accum.a, 1e-5);
    let oit_rgb = accum.rgb / weight;
    // Coverage: scale accum alpha, optional power curve, optional bias (paper: tune for content).
    let scaled = min(weight * oit_resolve.accum_alpha_scale, 1.0);
    var oit_alpha = pow(scaled, oit_resolve.resolve_opacity_power) + oit_resolve.opacity_bias;
    oit_alpha = clamp(oit_alpha, 0.0, 1.0);
    // Over blend: out = oit_rgb * oit_alpha + scene * (1 - oit_alpha)
    let rgb = oit_rgb * oit_alpha + scene.rgb * (1.0 - oit_alpha);
    let alpha = oit_alpha + scene.a * (1.0 - oit_alpha);
    return vec4<f32>(rgb, alpha);
}
