// Fullscreen resolve for weighted blended OIT: resolve accum then composite over scene.
// accum: (sum(premul*weight), sum(weight)); resolve to rgb/weight, alpha = min(weight,1).
// Output: scene * (1 - oit_alpha) + oit_rgb * oit_alpha (over blend).
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var accum_tex: texture_2d<f32>;
@group(0) @binding(2) var resolve_sampler: sampler;

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
    let oit_alpha = min(weight, 1.0);
    // Over blend: out = oit_rgb * oit_alpha + scene * (1 - oit_alpha)
    let rgb = oit_rgb * oit_alpha + scene.rgb * (1.0 - oit_alpha);
    let alpha = oit_alpha + scene.a * (1.0 - oit_alpha);
    return vec4<f32>(rgb, alpha);
}
