use bevy::prelude::*;
use bevy_args::{Deserialize, Serialize, ValueEnum};

use crate::camera::GaussianCamera;
use crate::sort::SortMode;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize)]
pub enum DrawMode {
    #[default]
    All,
    Selected,
    HighlightSelected,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize, ValueEnum,
)]
pub enum GaussianMode {
    Gaussian2d,
    #[default]
    Gaussian3d,
    Gaussian4d,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize, ValueEnum,
)]
pub enum PlaybackMode {
    Loop,
    Once,
    Sin,
    #[default]
    Still,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize, ValueEnum,
)]
pub enum RasterizeMode {
    Classification,
    #[default]
    Color,
    Depth,
    Normal,
    OpticalFlow,
    Position,
    Velocity,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize)]
pub enum GaussianColorSpace {
    #[default]
    SrgbRec709Display,
    LinRec709Display,
}

/// Screen-space footprint for each splat: axis-aligned quad vs oriented quad (ellipse axes).
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    PartialEq,
    Reflect,
    Serialize,
    Deserialize,
    ValueEnum,
)]
pub enum GaussianBounds {
    /// Axis-aligned square in NDC (max of ellipse axes).
    Aabb,
    /// Rotated quad aligned with projected covariance axes (default).
    #[default]
    Obb,
    /// Triangle strip with three vertices (less overdraw than a quad).
    Triangle,
}

impl GaussianBounds {
    /// Vertices per splat instance (`draw` / indirect `vertex_count`).
    #[inline]
    pub const fn splat_vertex_count(self) -> u32 {
        match self {
            Self::Triangle => 3,
            Self::Aabb | Self::Obb => 4,
        }
    }
}

// TODO: breakdown into components
#[derive(Component, Clone, Debug, Reflect, Serialize, Deserialize)]
#[reflect(Component, Default)]
#[serde(default)]
pub struct CloudSettings {
    pub bounds: GaussianBounds,
    /// Cap splats sorted & drawn per frame (`min(len, budget)`). `0` = use full cloud.
    ///
    /// Only indices `0 .. count` participate; training order is arbitrary — for quality at a cap,
    /// prefer offline LOD / decimation or a future GPU selection pass.
    pub splat_render_budget: u32,
    /// Multiplier on the screen-space Gaussian cutoff (smaller → tighter quads, less overdraw).
    pub quad_cutoff_scale: f32,
    pub global_opacity: f32,
    pub global_scale: f32,
    pub opacity_adaptive_radius: bool,
    pub visualize_bounding_box: bool,
    pub sort_mode: SortMode,
    pub draw_mode: DrawMode,
    pub gaussian_mode: GaussianMode,
    pub playback_mode: PlaybackMode,
    pub rasterize_mode: RasterizeMode,
    pub color_space: GaussianColorSpace,
    pub num_classes: usize,
    pub time: f32,
    pub time_scale: f32,
    pub time_start: f32,
    pub time_stop: f32,
    pub breathing_amplitude: f32,
    pub breathing_speed: f32,
    pub wave_amplitude: f32,
    pub wave_speed: f32,
    pub wave_frequency: f32,
    pub wave_direction: Vec3,
    pub pulse_amplitude: f32,
    pub pulse_speed: f32,
    pub pulse_frequency: f32,
    pub pulse_origin: Vec3,
    pub jitter_amplitude: f32,
    pub jitter_speed: f32,
    pub sparkle_amplitude: f32,
    pub sparkle_speed: f32,
    /// When true, strips entity rotation from covariance so splats stay camera-aligned
    pub billboard: bool,
    /// Render every Nth particle (1 = all, 2 = half, 4 = quarter). Higher = faster.
    pub subsample: u32,
    /// Skip particles with opacity below this threshold (0.0 = disabled)
    pub opacity_cutoff: f32,
    /// Skip particles beyond this distance from camera (0.0 = disabled)
    pub max_distance: f32,
    /// Clamp splat screen-space radius to this many pixels (0.0 = disabled)
    pub max_pixel_radius: f32,
    /// Override max_pixel_radius with per-splat distance-based clamping (reduces overdraw at range)
    pub dynamic_lod_radius: bool,
    /// Scale factor for dynamic LOD: base max radius (pixels) ≈ `dynamic_lod_scale / distance_to_camera`
    pub dynamic_lod_scale: f32,
    /// World-space reference extent for per-splat pixel-cap scaling (see [`Self::dynamic_lod_splat_extent_enabled`]).
    /// Only sent to the GPU when [`Self::dynamic_lod_splat_extent_enabled`] is true; otherwise treated as `0`.
    pub dynamic_lod_splat_reference: f32,
    /// When true, the shader uses [`Self::dynamic_lod_splat_reference`] to scale the pixel-radius cap by
    /// splat size. When false, behavior matches pre–per-splat scaling (distance / fixed cap only).
    pub dynamic_lod_splat_extent_enabled: bool,
    /// Distance at which LOD subsampling begins (0.0 = disabled)
    pub lod_near_distance: f32,
    /// Distance at which LOD subsampling reaches maximum
    pub lod_far_distance: f32,
    /// Maximum subsample factor at far distance
    pub lod_max_subsample: u32,
    /// Internal: set automatically when pulse triggers (elapsed_secs at trigger time)
    #[serde(skip)]
    pub pulse_start_time: f32,
}

impl CloudSettings {
    /// Reasonable cap for very dense captures when you prefer FPS over using every splat.
    pub const DENSE_DEFAULT_SPLAT_BUDGET: u32 = 2_000_000;
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self {
            bounds: GaussianBounds::default(),
            splat_render_budget: 0,
            quad_cutoff_scale: 1.0,
            global_opacity: 1.0,
            global_scale: 1.0,
            opacity_adaptive_radius: true,
            visualize_bounding_box: false,
            sort_mode: SortMode::default(),
            draw_mode: DrawMode::default(),
            gaussian_mode: GaussianMode::default(),
            rasterize_mode: RasterizeMode::default(),
            color_space: GaussianColorSpace::default(),
            num_classes: 1,
            playback_mode: PlaybackMode::default(),
            time: 0.0,
            time_scale: 1.0,
            time_start: 0.0,
            time_stop: 1.0,
            breathing_amplitude: 0.0,
            breathing_speed: 3.0,
            wave_amplitude: 0.0,
            wave_speed: 3.0,
            wave_frequency: 2.0,
            wave_direction: Vec3::Y,
            pulse_amplitude: 0.0,
            pulse_speed: 8.0,
            pulse_frequency: 3.0,
            pulse_origin: Vec3::ZERO,
            jitter_amplitude: 0.0,
            jitter_speed: 5.0,
            sparkle_amplitude: 0.0,
            sparkle_speed: 8.0,
            billboard: false,
            subsample: 1,
            opacity_cutoff: 0.0,
            max_distance: 0.0,
            max_pixel_radius: 0.0,
            dynamic_lod_radius: false,
            dynamic_lod_scale: 500.0,
            dynamic_lod_splat_reference: 0.0,
            dynamic_lod_splat_extent_enabled: false,
            lod_near_distance: 0.0,
            lod_far_distance: 50.0,
            lod_max_subsample: 8,
            pulse_start_time: 0.0,
        }
    }
}

#[derive(Default)]
pub struct SettingsPlugin;
impl Plugin for SettingsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<GaussianBounds>();
        app.register_type::<CloudSettings>();
        app.register_type::<DrawMode>();
        app.register_type::<GaussianMode>();
        app.register_type::<PlaybackMode>();
        app.register_type::<RasterizeMode>();
        app.register_type::<GaussianColorSpace>();

        app.add_systems(Update, (playback_update, effect_pulse_trigger, lod_update));
    }
}

fn playback_update(time: Res<Time>, mut query: Query<(&mut CloudSettings,)>) {
    for (mut settings,) in query.iter_mut() {
        if settings.time_scale == 0.0 {
            continue;
        }

        // bail condition
        match settings.playback_mode {
            PlaybackMode::Loop => {}
            PlaybackMode::Once => {
                if settings.time >= settings.time_stop {
                    continue;
                }
            }
            PlaybackMode::Sin => {}
            PlaybackMode::Still => {
                continue;
            }
        }

        // forward condition
        match settings.playback_mode {
            PlaybackMode::Loop | PlaybackMode::Once => {
                settings.time += time.delta_secs() * settings.time_scale;
            }
            PlaybackMode::Sin => {
                let theta = settings.time_scale * time.elapsed_secs();
                let y = (theta * 2.0 * std::f32::consts::PI).sin();
                settings.time = settings.time_start
                    + (settings.time_stop - settings.time_start) * (y + 1.0) / 2.0;
            }
            PlaybackMode::Still => {}
        }

        // reset condition
        match settings.playback_mode {
            PlaybackMode::Loop => {
                if settings.time > settings.time_stop {
                    settings.time = settings.time_start;
                }
            }
            PlaybackMode::Once => {}
            PlaybackMode::Sin => {}
            PlaybackMode::Still => {}
        }
    }
}

fn effect_pulse_trigger(time: Res<Time>, mut query: Query<&mut CloudSettings, Changed<CloudSettings>>) {
    for mut settings in query.iter_mut() {
        if settings.pulse_amplitude > 0.0 && settings.pulse_start_time == 0.0 {
            settings.bypass_change_detection().pulse_start_time = time.elapsed_secs();
        }
    }
}

fn lod_update(
    cameras: Query<&GlobalTransform, With<GaussianCamera>>,
    mut clouds: Query<(&GlobalTransform, &mut CloudSettings)>,
) {
    for (cloud_transform, mut settings) in clouds.iter_mut() {
        if settings.lod_near_distance <= 0.0 {
            continue;
        }

        let min_dist = cameras
            .iter()
            .map(|ct| ct.translation().distance(cloud_transform.translation()))
            .reduce(f32::min)
            .unwrap_or(0.0);

        let t = ((min_dist - settings.lod_near_distance)
            / (settings.lod_far_distance - settings.lod_near_distance))
            .clamp(0.0, 1.0);

        let subsample = 1 + (t * (settings.lod_max_subsample - 1) as f32) as u32;
        settings.bypass_change_detection().subsample = subsample;
    }
}
