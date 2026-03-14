// Fullscreen resolve for weighted blended OIT: accum.rgb/weight, alpha = min(weight, 1)
@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var accum_sampler: sampler;

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
    let accum = textureSample(accum_tex, accum_sampler, input.uv);
    let weight = max(accum.a, 1e-5);
    let rgb = accum.rgb / weight;
    let alpha = min(weight, 1.0);
    return vec4<f32>(rgb, alpha);
}
