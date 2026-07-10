//! Planar half-space clipping for gaussian clouds.
//!
//! Spawn a cutter as a child of a cloud (or target it with [`CutsGaussianCloud`]) and animate
//! its [`Transform`]. The cutter's world pose becomes the clip plane each frame.
//!
//! Clipping is applied per-fragment in the splat shader (not just gaussian centers), so
//! billboards straddling the plane are sliced cleanly.
//!
//! ```ignore
//! commands.entity(cloud).with_children(|c| {
//!     c.spawn((
//!         GaussianPlaneCutter::default(),
//!         Transform::from_xyz(0.0, cut_height, 0.0),
//!     ));
//! });
//! ```

use bevy::prelude::*;
use bevy_args::{Deserialize, Serialize};

/// Local axis of the cutter transform used as the plane normal.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Default)]
pub enum PlaneNormalAxis {
    X,
    #[default]
    Y,
    Z,
}

impl PlaneNormalAxis {
    pub fn as_vec3(self) -> Vec3 {
        match self {
            Self::X => Vec3::X,
            Self::Y => Vec3::Y,
            Self::Z => Vec3::Z,
        }
    }
}

/// Marker / settings on a cutter entity whose [`Transform`] defines the clip plane.
///
/// Target the cloud with [`CutsGaussianCloud`], or parent the cutter under the cloud via
/// [`ChildOf`] / `with_children` (parent is used when the relationship is absent).
#[derive(Component, Clone, Debug, Reflect, Serialize, Deserialize)]
#[reflect(Component, Default)]
#[serde(default)]
pub struct GaussianPlaneCutter {
    pub enabled: bool,
    /// When true, keep splats with signed distance `dot(n, p - p0) >= 0`.
    /// When false, keep the opposite half-space (typical for a rising floor cut).
    pub keep_positive_side: bool,
    /// Cutter-local axis mapped through the cutter rotation to world normal.
    pub normal_axis: PlaneNormalAxis,
}

impl Default for GaussianPlaneCutter {
    fn default() -> Self {
        Self {
            enabled: true,
            keep_positive_side: false,
            normal_axis: PlaneNormalAxis::Y,
        }
    }
}

/// Relationship: this cutter clips the referenced gaussian cloud entity.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
#[relationship(relationship_target = GaussianPlaneCutters)]
pub struct CutsGaussianCloud(#[entities] pub Entity);

impl CutsGaussianCloud {
    pub fn cloud(&self) -> Entity {
        self.0
    }
}

/// All plane cutters targeting a cloud (maintained by the relationship).
#[derive(Component, Debug)]
#[relationship_target(relationship = CutsGaussianCloud)]
pub struct GaussianPlaneCutters(Vec<Entity>);

impl GaussianPlaneCutters {
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.0.iter().copied()
    }
}

/// Resolved world-space plane on the **cloud** entity, written by
/// [`sync_gaussian_plane_cuts`] and read during render extract.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component, Default)]
pub struct GaussianPlaneCut {
    pub enabled: bool,
    pub point: Vec3,
    pub normal: Vec3,
    pub keep_positive_side: bool,
}

impl Default for GaussianPlaneCut {
    fn default() -> Self {
        Self {
            enabled: false,
            point: Vec3::ZERO,
            normal: Vec3::Y,
            keep_positive_side: false,
        }
    }
}

impl GaussianPlaneCut {
    /// Pack into cloud uniform fields: `plane_point.w` = enabled, `plane_normal.w` = ±1 keep side.
    pub fn to_uniform_fields(self) -> (Vec4, Vec4) {
        if !self.enabled {
            return (Vec4::ZERO, Vec3::Y.extend(1.0));
        }
        let keep = if self.keep_positive_side { 1.0 } else { -1.0 };
        (
            self.point.extend(1.0),
            self.normal.normalize_or_zero().extend(keep),
        )
    }

    /// True when `world_pos` should be discarded by this plane.
    pub fn discards(&self, world_pos: Vec3) -> bool {
        if !self.enabled {
            return false;
        }
        let n = self.normal.normalize_or_zero();
        if n == Vec3::ZERO {
            return false;
        }
        let d = n.dot(world_pos - self.point);
        if self.keep_positive_side {
            d < 0.0
        } else {
            d > 0.0
        }
    }
}

#[derive(Default)]
pub struct PlaneCutPlugin;

impl Plugin for PlaneCutPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<GaussianPlaneCutter>()
            .register_type::<GaussianPlaneCut>()
            .register_type::<PlaneNormalAxis>()
            .add_systems(
                PostUpdate,
                sync_gaussian_plane_cuts.after(TransformSystems::Propagate),
            );
    }
}

/// Resolve cutter transforms onto each targeted cloud's [`GaussianPlaneCut`].
///
/// When multiple cutters target one cloud, the last one in query order wins.
pub fn sync_gaussian_plane_cuts(
    mut commands: Commands,
    cutters: Query<(
        &GaussianPlaneCutter,
        &GlobalTransform,
        Option<&CutsGaussianCloud>,
        Option<&ChildOf>,
    )>,
    mut clouds: Query<(Entity, &mut GaussianPlaneCut)>,
) {
    let mut updates: Vec<(Entity, GaussianPlaneCut)> = Vec::new();
    let mut touched = bevy::platform::collections::HashSet::<Entity>::default();

    for (cutter, global, cuts, child_of) in &cutters {
        let Some(cloud) = cuts
            .map(|c| c.cloud())
            .or_else(|| child_of.map(|c| c.parent()))
        else {
            continue;
        };

        let (_, rotation, translation) = global.to_scale_rotation_translation();
        let normal = (rotation * cutter.normal_axis.as_vec3()).normalize_or_zero();
        touched.insert(cloud);
        updates.push((
            cloud,
            GaussianPlaneCut {
                enabled: cutter.enabled && normal != Vec3::ZERO,
                point: translation,
                normal,
                keep_positive_side: cutter.keep_positive_side,
            },
        ));
    }

    for (entity, mut cut) in &mut clouds {
        if !touched.contains(&entity) && cut.enabled {
            cut.enabled = false;
        }
    }

    for (cloud, resolved) in updates {
        if let Ok((_, mut cut)) = clouds.get_mut(cloud) {
            *cut = resolved;
        } else {
            commands.entity(cloud).try_insert(resolved);
        }
    }
}
