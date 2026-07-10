#define_import_path bevy_gaussian_splatting::transform

#import bevy_gaussian_splatting::bindings::{gaussian_uniforms, view}

fn world_to_clip(world_pos: vec3<f32>) -> vec4<f32> {
    let homogenous_pos = view.unjittered_clip_from_world * vec4<f32>(world_pos, 1.0);
    return homogenous_pos / (homogenous_pos.w + 0.000000001);
}

fn world_to_clip_homogeneous(world_pos: vec3<f32>) -> vec4<f32> {
    return view.unjittered_clip_from_world * vec4<f32>(world_pos, 1.0);
}

/// Unproject an NDC xy offset at the depth of `world_pos` back to world space.
fn ndc_offset_to_world(world_pos: vec3<f32>, ndc_offset: vec2<f32>) -> vec3<f32> {
    let clip = world_to_clip_homogeneous(world_pos);
    let w = clip.w + 0.000000001;
    let ndc = clip.xyz / w;
    let offset_clip = vec4<f32>((ndc.xy + ndc_offset) * w, clip.z, clip.w);
    let world = view.world_from_clip * offset_clip;
    return world.xyz / world.w;
}

fn in_frustum(clip_space_pos: vec3<f32>) -> bool {
    return abs(clip_space_pos.x) < 1.1
        && abs(clip_space_pos.y) < 1.1
        && abs(clip_space_pos.z - 0.5) < 0.5;
}

/// Discard when the world-space point is on the clipped side of the cloud plane cut.
fn discarded_by_plane_cut(world_pos: vec3<f32>) -> bool {
    if (gaussian_uniforms.plane_point.w < 0.5) {
        return false;
    }
    let n = gaussian_uniforms.plane_normal.xyz;
    let d = dot(n, world_pos - gaussian_uniforms.plane_point.xyz);
    // plane_normal.w > 0 → keep d >= 0; else keep d <= 0
    if (gaussian_uniforms.plane_normal.w > 0.0) {
        return d < 0.0;
    }
    return d > 0.0;
}
