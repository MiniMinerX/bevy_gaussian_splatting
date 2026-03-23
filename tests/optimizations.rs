//! Correctness tests for rendering optimizations.

use bevy::math::Vec3A;
use bevy_gaussian_splatting::{
    CloudSettings,
    sort::SortConfig,
};

#[test]
fn test_max_pixel_radius_default_disabled() {
    let settings = CloudSettings::default();
    assert_eq!(settings.max_pixel_radius, 0.0, "max_pixel_radius should default to 0.0 (disabled)");
}

#[test]
fn test_max_pixel_radius_configurable() {
    let settings = CloudSettings {
        max_pixel_radius: 50.0,
        ..Default::default()
    };
    assert_eq!(settings.max_pixel_radius, 50.0);
}

#[test]
fn test_sort_movement_threshold_default() {
    let config = SortConfig::default();
    assert_eq!(config.movement_threshold, 0.01);
}

#[test]
fn test_sort_movement_threshold_suppresses_small_moves() {
    let config = SortConfig {
        movement_threshold: 100.0,
        ..Default::default()
    };

    let last_pos = Vec3A::new(0.0, 0.0, 0.0);
    let new_pos = Vec3A::new(0.01, 0.0, 0.0);
    let moved = last_pos.distance(new_pos) > config.movement_threshold;
    assert!(!moved, "small movement should not trigger sort with high threshold");
}

#[test]
fn test_sort_movement_threshold_allows_large_moves() {
    let config = SortConfig {
        movement_threshold: 0.01,
        ..Default::default()
    };

    let last_pos = Vec3A::new(0.0, 0.0, 0.0);
    let new_pos = Vec3A::new(1.0, 0.0, 0.0);
    let moved = last_pos.distance(new_pos) > config.movement_threshold;
    assert!(moved, "large movement should trigger sort");
}

#[test]
fn test_lod_defaults() {
    let settings = CloudSettings::default();
    assert_eq!(settings.lod_near_distance, 0.0, "LOD disabled by default");
    assert_eq!(settings.lod_far_distance, 50.0);
    assert_eq!(settings.lod_max_subsample, 8);
}

#[test]
fn test_lod_subsample_computation() {
    let near = 10.0_f32;
    let far = 50.0_f32;
    let max_subsample = 8_u32;

    let dist = 10.0_f32;
    let t = ((dist - near) / (far - near)).clamp(0.0, 1.0);
    let subsample = 1 + (t * (max_subsample - 1) as f32) as u32;
    assert_eq!(subsample, 1, "at near distance, subsample should be 1");

    let dist = 50.0_f32;
    let t = ((dist - near) / (far - near)).clamp(0.0, 1.0);
    let subsample = 1 + (t * (max_subsample - 1) as f32) as u32;
    assert_eq!(subsample, max_subsample, "at far distance, subsample should be max");

    let dist = 30.0_f32;
    let t = ((dist - near) / (far - near)).clamp(0.0, 1.0);
    let subsample = 1 + (t * (max_subsample - 1) as f32) as u32;
    assert!(subsample > 1 && subsample < max_subsample, "at mid distance, subsample should be between 1 and max");

    let dist = 100.0_f32;
    let t = ((dist - near) / (far - near)).clamp(0.0, 1.0);
    let subsample = 1 + (t * (max_subsample - 1) as f32) as u32;
    assert_eq!(subsample, max_subsample, "beyond far distance, subsample should be max");
}

#[test]
fn test_cloud_settings_serde_roundtrip() {
    let settings = CloudSettings {
        max_pixel_radius: 42.0,
        lod_near_distance: 5.0,
        lod_far_distance: 100.0,
        lod_max_subsample: 16,
        ..Default::default()
    };

    let json = serde_json::to_string(&settings).expect("serialize");
    let deserialized: CloudSettings = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.max_pixel_radius, 42.0);
    assert_eq!(deserialized.lod_near_distance, 5.0);
    assert_eq!(deserialized.lod_far_distance, 100.0);
    assert_eq!(deserialized.lod_max_subsample, 16);
}
