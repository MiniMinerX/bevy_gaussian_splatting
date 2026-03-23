//! Rendering benchmark for gaussian splatting optimizations.
//!
//! Measures frame times across a matrix of particle counts, global scales,
//! and optimization settings. Outputs CSV to stdout.
//!
//! Usage: cargo run --bin bench_render --no-default-features --features "headless,perftest"

use bevy::{
    app::ScheduleRunnerPlugin,
    camera::RenderTarget,
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::render_resource::{Extent3d, TextureFormat, TextureUsages},
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, GaussianSplattingPlugin, PlanarGaussian3d,
    PlanarGaussian3dHandle, random_gaussians_3d_seeded,
    sort::SortConfig,
};
use std::time::{Duration, Instant};

const WARMUP_FRAMES: u32 = 40;
const MEASURE_FRAMES: u32 = 200;
const RESOLUTION: (u32, u32) = (1920, 1080);
const CAMERA_POS: Vec3 = Vec3::new(0.0, 1.5, 5.0);

#[derive(Clone)]
struct BenchScenario {
    label: String,
    particle_count: usize,
    global_scale: f32,
    max_pixel_radius: f32,
    movement_threshold: f32,
}

#[derive(Resource)]
struct BenchState {
    scenarios: Vec<BenchScenario>,
    current_scenario: usize,
    frame_count: u32,
    frame_times: Vec<Duration>,
    last_frame_time: Option<Instant>,
    results: Vec<BenchResult>,
    phase: BenchPhase,
}

#[derive(Clone, Copy, PartialEq)]
enum BenchPhase {
    Setup,
    Warmup,
    Measure,
    Done,
}

struct BenchResult {
    label: String,
    particle_count: usize,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    fps: f64,
}

fn build_scenarios() -> Vec<BenchScenario> {
    let particle_counts = [1_000, 10_000, 100_000, 500_000];
    let global_scales = [0.5, 1.0, 2.0, 4.0];
    let mut scenarios = Vec::new();

    for &count in &particle_counts {
        for &scale in &global_scales {
            scenarios.push(BenchScenario {
                label: format!("baseline_n{}_s{}", count, scale),
                particle_count: count,
                global_scale: scale,
                max_pixel_radius: 0.0,
                movement_threshold: 0.01,
            });
        }
    }

    for &count in &particle_counts {
        scenarios.push(BenchScenario {
            label: format!("max_px_radius_50_n{}_s4", count),
            particle_count: count,
            global_scale: 4.0,
            max_pixel_radius: 50.0,
            movement_threshold: 0.01,
        });
    }

    for &count in &particle_counts {
        scenarios.push(BenchScenario {
            label: format!("sort_thresh_1.0_n{}_s1", count),
            particle_count: count,
            global_scale: 1.0,
            max_pixel_radius: 0.0,
            movement_threshold: 1.0,
        });
    }

    scenarios
}

#[derive(Component)]
struct BenchCloud;

fn main() {
    let scenarios = build_scenarios();
    eprintln!("Running {} benchmark scenarios", scenarios.len());
    println!("scenario,particle_count,mean_ms,median_ms,p95_ms,fps");

    App::new()
        .insert_resource(BenchState {
            scenarios,
            current_scenario: 0,
            frame_count: 0,
            frame_times: Vec::with_capacity(MEASURE_FRAMES as usize),
            last_frame_time: None,
            results: Vec::new(),
            phase: BenchPhase::Setup,
        })
        .insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)))
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 120.0,
        )))
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup_camera)
        .add_systems(Update, bench_driver)
        .run();
}

fn setup_camera(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = Extent3d {
        width: RESOLUTION.0,
        height: RESOLUTION.1,
        ..default()
    };
    let mut render_target_image =
        Image::new_target_texture(size.width, size.height, TextureFormat::bevy_default(), None);
    render_target_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let render_target_handle = images.add(render_target_image);

    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        RenderTarget::Image(render_target_handle.into()),
        Transform::from_translation(CAMERA_POS),
        Tonemapping::None,
        GaussianCamera::default(),
    ));
}

fn bench_driver(
    mut commands: Commands,
    mut state: ResMut<BenchState>,
    mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
    mut sort_config: ResMut<SortConfig>,
    mut app_exit: MessageWriter<AppExit>,
    clouds: Query<Entity, With<BenchCloud>>,
) {
    match state.phase {
        BenchPhase::Setup => {
            for entity in clouds.iter() {
                commands.entity(entity).despawn();
            }

            if state.current_scenario >= state.scenarios.len() {
                state.phase = BenchPhase::Done;
                return;
            }

            let scenario = state.scenarios[state.current_scenario].clone();
            eprintln!(
                "[{}/{}] {}",
                state.current_scenario + 1,
                state.scenarios.len(),
                scenario.label
            );

            sort_config.movement_threshold = scenario.movement_threshold;

            let cloud =
                gaussian_assets.add(random_gaussians_3d_seeded(scenario.particle_count, 42));

            commands.spawn((
                PlanarGaussian3dHandle(cloud),
                CloudSettings {
                    global_scale: scenario.global_scale,
                    max_pixel_radius: scenario.max_pixel_radius,
                    ..default()
                },
                Name::new("bench_cloud"),
                Transform::IDENTITY,
                BenchCloud,
            ));

            state.frame_count = 0;
            state.frame_times.clear();
            state.last_frame_time = None;
            state.phase = BenchPhase::Warmup;
        }
        BenchPhase::Warmup => {
            state.frame_count += 1;
            if state.frame_count >= WARMUP_FRAMES {
                state.frame_count = 0;
                state.last_frame_time = Some(Instant::now());
                state.phase = BenchPhase::Measure;
            }
        }
        BenchPhase::Measure => {
            let now = Instant::now();
            if let Some(last) = state.last_frame_time {
                state.frame_times.push(now - last);
            }
            state.last_frame_time = Some(now);
            state.frame_count += 1;

            if state.frame_count >= MEASURE_FRAMES {
                let scenario = &state.scenarios[state.current_scenario];
                let result = compute_result(
                    &scenario.label,
                    scenario.particle_count,
                    &state.frame_times,
                );
                println!(
                    "{},{},{:.3},{:.3},{:.3},{:.1}",
                    result.label,
                    result.particle_count,
                    result.mean_ms,
                    result.median_ms,
                    result.p95_ms,
                    result.fps
                );
                state.results.push(result);
                state.current_scenario += 1;
                state.phase = BenchPhase::Setup;
            }
        }
        BenchPhase::Done => {
            eprintln!("Benchmark complete. {} scenarios measured.", state.results.len());
            app_exit.write(AppExit::Success);
        }
    }
}

fn compute_result(label: &str, particle_count: usize, frame_times: &[Duration]) -> BenchResult {
    let mut times_ms: Vec<f64> = frame_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = times_ms.len();
    let mean_ms = times_ms.iter().sum::<f64>() / n as f64;
    let median_ms = if n % 2 == 0 {
        (times_ms[n / 2 - 1] + times_ms[n / 2]) / 2.0
    } else {
        times_ms[n / 2]
    };
    let p95_idx = ((n as f64) * 0.95) as usize;
    let p95_ms = times_ms[p95_idx.min(n - 1)];
    let fps = 1000.0 / mean_ms;

    BenchResult {
        label: label.to_string(),
        particle_count,
        mean_ms,
        median_ms,
        p95_ms,
        fps,
    }
}
