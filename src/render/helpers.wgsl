#define_import_path bevy_gaussian_splatting::helpers

#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
}

fn cov2d(
    position: vec3<f32>,
    cov3d: array<f32, 6>,
) -> vec3<f32> {
    let Vrk = mat3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    var t = view.view_from_world * vec4<f32>(position, 1.0);

    // Use main_pass_viewport dimensions so focal length and pixel radii match the actual
    // rasterized main pass (Bevy: viewport can differ from main_pass when resolution override /
    // upscaling / scaled rendering is used).
    let focal = vec2<f32>(
        view.clip_from_view[0].x * view.main_pass_viewport.z,
        view.clip_from_view[1].y * view.main_pass_viewport.w,
    );

    let s = 1.0 / (t.z * t.z);
    let J = mat3x3(
        focal.x / t.z, 0.0, -(focal.x * t.x) * s,
        0.0, -focal.y / t.z, (focal.y * t.y) * s,
        0.0, 0.0, 0.0,
    );

    let W = transpose(
        mat3x3<f32>(
            view.view_from_world[0].xyz,
            view.view_from_world[1].xyz,
            view.view_from_world[2].xyz,
        )
    );

    let T = W * J;

    var cov = transpose(T) * transpose(Vrk) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}

fn get_bounding_box_clip(
    cov2d: vec3<f32>,
    direction: vec2<f32>,
    cutoff: f32,
    cam_distance: f32,
) -> vec4<f32> {
    // return vec4<f32>(offset, uv);

    var effective_max_radius = gaussian_uniforms.max_pixel_radius;
    if gaussian_uniforms.dynamic_lod_radius > 0u {
        effective_max_radius = gaussian_uniforms.dynamic_lod_scale / max(cam_distance, 0.01);
    }

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let trace = cov2d.x + cov2d.z;
    let mid = 0.5 * trace;
    let discriminant = max(0.0, mid * mid - det);

    let term = sqrt(discriminant);

    let lambda1 = mid + term;
    let lambda2 = max(mid - term, 0.0);

    let x_axis_length = sqrt(lambda1);
    let y_axis_length = sqrt(lambda2);

#ifdef USE_AABB
    var radius_px = cutoff * max(x_axis_length, y_axis_length);
    if effective_max_radius > 0.0 {
        radius_px = min(radius_px, effective_max_radius);
    }
    let radius_ndc = vec2<f32>(
        radius_px / view.main_pass_viewport.zw,
    );

    return vec4<f32>(
        radius_ndc * direction,
        radius_px * direction,
    );
#endif

#ifdef USE_OBB

    let a = (cov2d.x - cov2d.z) * (cov2d.x - cov2d.z);
    let b = sqrt(a + 4.0 * cov2d.y * cov2d.y);
    var major_radius = sqrt((cov2d.x + cov2d.z + b) * 0.5);
    var minor_radius = sqrt((cov2d.x + cov2d.z - b) * 0.5);
    if effective_max_radius > 0.0 {
        major_radius = min(major_radius, effective_max_radius);
        minor_radius = min(minor_radius, effective_max_radius);
    }

    let bounds = cutoff * vec2<f32>(
        major_radius,
        minor_radius,
    );

    let eigvec1 = normalize(vec2<f32>(
        -cov2d.y,
        lambda1 - cov2d.x,
    ));
    let eigvec2 = vec2<f32>(
        eigvec1.y,
        -eigvec1.x
    );

    let rotation_matrix = transpose(
        mat2x2(
            eigvec1,
            eigvec2,
        )
    );

    let scaled_vertex = direction * bounds;
    let rotated_vertex = scaled_vertex * rotation_matrix;

    let scaling_factor = 1.0 / view.main_pass_viewport.zw;
    let ndc_vertex = rotated_vertex * scaling_factor;

    return vec4<f32>(
        ndc_vertex,
        rotated_vertex,
    );
#endif
}

fn intrinsic_matrix() -> mat3x4<f32> {
    let focal = vec2<f32>(
        view.clip_from_view[0].x * view.main_pass_viewport.z / 2.0,
        view.clip_from_view[1].y * view.main_pass_viewport.w / 2.0,
    );

    let Ks = mat3x4<f32>(
        vec4<f32>(focal.x, 0.0, 0.0, (view.main_pass_viewport.z - 1.0) / 2.0),
        vec4<f32>(0.0, focal.y, 0.0, (view.main_pass_viewport.w - 1.0) / 2.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );

    return Ks;
}

fn get_rotation_matrix(
    rotation: vec4<f32>,
) -> mat3x3<f32> {
    let r = rotation.x;
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;

    return mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - r * z),
        2.0 * (x * z + r * y),

        2.0 * (x * y + r * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - r * x),

        2.0 * (x * z - r * y),
        2.0 * (y * z + r * x),
        1.0 - 2.0 * (x * x + y * y),
    );
}

fn get_scale_matrix(
    scale: vec3<f32>,
) -> mat3x3<f32> {
    return mat3x3<f32>(
        scale.x * gaussian_uniforms.global_scale, 0.0, 0.0,
        0.0, scale.y * gaussian_uniforms.global_scale, 0.0,
        0.0, 0.0, scale.z * gaussian_uniforms.global_scale,
    );
}

fn hash_u(n: u32) -> f32 {
    var x = n;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return f32(x) / 4294967295.0;
}

fn value_noise(t: f32, seed: u32) -> f32 {
    let i = u32(floor(t)) + seed;
    let f = fract(t);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(hash_u(i), hash_u(i + 1u), u) * 2.0 - 1.0;
}

fn fbm_noise(t: f32, seed: u32) -> f32 {
    return value_noise(t, seed) * 0.6
         + value_noise(t * 2.0, seed + 7919u) * 0.25
         + value_noise(t * 4.0, seed + 15973u) * 0.15;
}
