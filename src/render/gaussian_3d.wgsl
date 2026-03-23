#define_import_path bevy_gaussian_splatting::gaussian_3d

#ifdef GAUSSIAN_3D
#import bevy_gaussian_splatting::bindings::{
    globals,
    view,
    gaussian_uniforms,
}
#import bevy_gaussian_splatting::helpers::{
    cov2d,
    get_rotation_matrix,
    get_scale_matrix,
    hash_u,
    fbm_noise,
}

#ifdef PACKED
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::packed::{
            get_cov3d,
        }
    #else
        #import bevy_gaussian_splatting::packed::{
            get_rotation,
            get_scale,
        }
    #endif
#else ifdef BUFFER_STORAGE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::planar::{
            get_cov3d,
        }
    #else
        #import bevy_gaussian_splatting::planar::{
            get_rotation,
            get_scale,
        }
    #endif
#else ifdef BUFFER_TEXTURE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::texture::{
            get_cov3d,
        }
    #else
        #import bevy_gaussian_splatting::texture::{
            get_rotation,
            get_scale,
        }
    #endif
#endif

fn compute_cov3d(scale: vec3<f32>, rotation: vec4<f32>) -> array<f32, 6> {
    let S = get_scale_matrix(scale);

    var T: mat3x3<f32>;
    if gaussian_uniforms.billboard != 0u {
        let sx = length(gaussian_uniforms.transform[0].xyz);
        let sy = length(gaussian_uniforms.transform[1].xyz);
        let sz = length(gaussian_uniforms.transform[2].xyz);
        T = mat3x3<f32>(
            vec3<f32>(sx, 0.0, 0.0),
            vec3<f32>(0.0, sy, 0.0),
            vec3<f32>(0.0, 0.0, sz),
        );
    } else {
        T = mat3x3<f32>(
            gaussian_uniforms.transform[0].xyz,
            gaussian_uniforms.transform[1].xyz,
            gaussian_uniforms.transform[2].xyz,
        );
    }

    let R = get_rotation_matrix(rotation);

    let M = S * R;
    let Sigma = transpose(M) * M;
    let TS = T * Sigma * transpose(T);

    return array<f32, 6>(
        TS[0][0],
        TS[0][1],
        TS[0][2],
        TS[1][1],
        TS[1][2],
        TS[2][2],
    );
}

fn compute_cov2d_3dgs(
    position: vec3<f32>,
    index: u32,
) -> vec3<f32> {
#ifdef PRECOMPUTE_COVARIANCE_3D
    let cov3d = get_cov3d(index);
#else
    let rotation = get_rotation(index);
    var scale = get_scale(index);

    var scale_mod = 0.0;

    scale_mod += gaussian_uniforms.breathing_amplitude
        * sin(globals.time * gaussian_uniforms.breathing_speed);

    let wave_phase = dot(position, gaussian_uniforms.wave_direction.xyz)
        * gaussian_uniforms.wave_frequency
        + globals.time * gaussian_uniforms.wave_speed;
    scale_mod += gaussian_uniforms.wave_amplitude * sin(wave_phase);

    let pulse_start = gaussian_uniforms.pulse_origin.w;
    let pulse_elapsed = globals.time - pulse_start;
    if pulse_elapsed > 0.0 && gaussian_uniforms.pulse_amplitude > 0.0 {
        let dist = length(position - gaussian_uniforms.pulse_origin.xyz);
        let p_phase = dist * gaussian_uniforms.pulse_frequency
            - pulse_elapsed * gaussian_uniforms.pulse_speed;
        scale_mod += gaussian_uniforms.pulse_amplitude
            * sin(p_phase) * exp(-0.5 * pulse_elapsed);
    }

    let sparkle_seed = index * 5u + 31337u;
    let sparkle_t = globals.time * gaussian_uniforms.sparkle_speed + hash_u(sparkle_seed) * 1000.0;
    let sparkle_val = fbm_noise(sparkle_t, sparkle_seed) * 0.5 + 0.5;
    scale_mod += gaussian_uniforms.sparkle_amplitude * sparkle_val;

    scale = scale * (1.0 + scale_mod);

    let cov3d = compute_cov3d(scale, rotation);
#endif

    return cov2d(position, cov3d);
}

#endif  // GAUSSIAN_3D
