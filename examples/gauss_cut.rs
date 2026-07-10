use bevy::{
    app::AppExit, core_pipeline::tonemapping::Tonemapping, prelude::*,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, GaussianPlaneCutter, GaussianSplattingPlugin,
    PlanarGaussian3d, PlanarGaussian3dHandle, random_gaussians_3d, utils::setup_hooks,
};

const GAUSSIAN_COUNT: usize = 8_000;
const CUT_AMPLITUDE: f32 = 18.0;
const CUT_SPEED: f32 = 0.6;

#[derive(Component)]
struct AnimatedCutPlane;

fn main() {
    setup_hooks();

    App::new()
        .insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "gauss_cut".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (animate_cut_plane, esc_close))
        .run();
}

fn setup(
    mut commands: Commands,
    mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cloud = gaussian_assets.add(random_gaussians_3d(GAUSSIAN_COUNT));

    commands
        .spawn((
            Name::new("gaussian_cloud"),
            PlanarGaussian3dHandle(cloud),
            CloudSettings::default(),
            Transform::default(),
            Visibility::default(),
        ))
        .with_children(|parent| {
            parent.spawn((
                Name::new("cut_plane"),
                AnimatedCutPlane,
                GaussianPlaneCutter::default(),
                Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(25.0)))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgba(0.3, 0.8, 1.0, 0.25),
                    alpha_mode: AlphaMode::Blend,
                    unlit: true,
                    cull_mode: None,
                    ..default()
                })),
                Transform::default(),
            ));
        });

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 8.0, 45.0)),
        Tonemapping::None,
        PanOrbitCamera {
            allow_upside_down: true,
            ..default()
        },
        GaussianCamera::default(),
    ));
}

fn animate_cut_plane(time: Res<Time>, mut query: Query<&mut Transform, With<AnimatedCutPlane>>) {
    for mut transform in &mut query {
        transform.translation.y = (time.elapsed_secs() * CUT_SPEED).sin() * CUT_AMPLITUDE;
    }
}

fn esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}
