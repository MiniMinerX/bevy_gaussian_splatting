//! Order-independent transparency (OIT) render path: accum texture + fullscreen resolve.
//! Avoids per-frame sorting; uses weighted blended OIT (McGuire & Bavoil) with Rgba32Float accum.

use std::any::TypeId;
use std::collections::HashMap;

use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    core_pipeline::prepass::{MotionVectorPrepass, PreviousViewUniformOffset},
    ecs::query::QueryItem,
    prelude::*,
    render::{
        extract_component::{DynamicUniformIndex, ExtractComponent},
        render_asset::RenderAssets,
        render_graph::{
            Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode,
            ViewNodeRunner,
        },
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        view::{ViewTarget, ViewUniformOffset},
        Render, RenderApp, RenderStartup,
    },
};
use bevy_interleave::prelude::*;
use bytemuck;

use crate::gaussian::formats::{planar_3d::Gaussian3d, planar_4d::Gaussian4d};
#[cfg(feature = "buffer_storage")]
use crate::sort::SortEntry;
use crate::camera::GaussianCamera;
use crate::render::{
    self, CloudUniform, GaussianUniformBindGroups, GaussianViewBindGroup,
    PlanarStorageBindGroup, SortBindGroup, ViewOitItems,
};
use crate::sort::SortTrigger;
use bevy::render::view::{ExtractedView, ViewDepthTexture};

// #region agent log
#[cfg(debug_assertions)]
fn debug_log(hypothesis_id: &str, location: &str, message: &str, data: serde_json::Value) {
    use std::{
        fs::OpenOptions,
        io::Write,
        sync::atomic::{AtomicUsize, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };
    static LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
    if LOG_COUNT.fetch_add(1, Ordering::Relaxed) >= 200 {
        return;
    }
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    let payload = serde_json::json!({
        "sessionId": "a8c8d4",
        "runId": "pre-fix-1",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": timestamp
    });
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("debug-a8c8d4.log")
    {
        let _ = writeln!(file, "{payload}");
    }
}

#[cfg(not(debug_assertions))]
fn debug_log(_hypothesis_id: &str, _location: &str, _message: &str, _data: serde_json::Value) {}
// #endregion

// --- OIT settings (inspector-editable component) -----------------------------------------------
// Paper: McGuire & Bavoil, "Weighted Blended OIT", JCGT 2013. Depth weight tunes how much
// closer surfaces dominate; steeper curves (higher power) help high-opacity content.

/// Per-camera settings for weighted blended OIT. Add to a camera with [`GaussianCamera`]
/// when using [`SortMode::Oit`] to tune accumulation and resolve behavior in the inspector.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct OitSettings {
    /// Scale for depth-based weight. Weight = 1 / (1 + (view_depth * scale)^power).
    /// Higher values favor closer fragments more (default: `1.0`).
    pub depth_weight_scale: f32,
    /// Minimum fragment weight to avoid division issues in resolve (default: `1e-4`).
    pub min_weight: f32,
    /// Exponent for depth weight: 1 / (1 + (depth*scale)^power). Power > 1 (e.g. 2) gives
    /// steeper falloff so closer splats dominate more; paper suggests tuning for content (default: `1.0`).
    pub depth_weight_power: f32,
    /// Multiply per-fragment alpha before accumulation. > 1 makes splats more opaque (default: `1.0`).
    pub opacity_scale: f32,
    /// Resolve: multiply accumulated alpha by this before coverage. > 1 reduces transparency (default: `1.0`).
    pub accum_alpha_scale: f32,
    /// Resolve: oit_alpha = pow(saturate(accum.a), power). Power < 1 (e.g. 0.7) boosts low alphas (default: `1.0`).
    pub resolve_opacity_power: f32,
    /// Resolve: add this to oit_alpha before over-blend. Slight positive bias reduces see-through (default: `0.0`).
    pub opacity_bias: f32,
    /// If true, render accum at half resolution for ~4x less fill; resolve upscales (default: `false`).
    pub half_res: bool,
}

impl Default for OitSettings {
    fn default() -> Self {
        Self {
            depth_weight_scale: 1.0,
            min_weight: 1e-4,
            depth_weight_power: 1.0,
            opacity_scale: 1.0,
            // Default > 1 so resolved alpha isn't stuck too transparent (weighted blend often under-reports coverage).
            accum_alpha_scale: 2.5,
            // Power < 1 boosts low alphas so thin splats don't disappear.
            resolve_opacity_power: 0.85,
            // Small bias reduces see-through; tune in inspector if needed.
            opacity_bias: 0.03,
            // Half-res accum (~4x less fill) for better OIT performance vs radix.
            half_res: true,
        }
    }
}

impl ExtractComponent for OitSettings {
    type QueryData = &'static Self;
    type QueryFilter = With<Camera>;
    type Out = Self;

    fn extract_component(settings: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(*settings)
    }
}

/// GPU uniform for OIT settings (must match WGSL layout).
#[derive(Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct OitSettingsUniform {
    pub depth_weight_scale: f32,
    pub min_weight: f32,
    pub depth_weight_power: f32,
    pub opacity_scale: f32,
    pub accum_alpha_scale: f32,
    pub resolve_opacity_power: f32,
    pub opacity_bias: f32,
    pub _pad0: f32,
    pub half_res: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

impl Default for OitSettingsUniform {
    fn default() -> Self {
        let s = OitSettings::default();
        Self {
            depth_weight_scale: s.depth_weight_scale,
            min_weight: s.min_weight,
            depth_weight_power: s.depth_weight_power,
            opacity_scale: s.opacity_scale,
            accum_alpha_scale: s.accum_alpha_scale,
            resolve_opacity_power: s.resolve_opacity_power,
            opacity_bias: s.opacity_bias,
            _pad0: 0.0,
            half_res: s.half_res as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        }
    }
}

/// OIT clear pass: clear accum texture for every view that has OIT content.
/// Runs before accum so 3d/4d passes can use Load and never show stale content.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct OitClearLabel;

/// OIT accumulation pass: draw gaussians to accum texture (additive).
/// One label per format so we can order 3d before 4d (both use Load after clear).
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct OitAccumLabel3d;
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct OitAccumLabel4d;

/// OIT resolve pass: fullscreen resolve + composite over scene.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct OitResolveLabel;

/// Per-view OIT accumulation texture (Rgba32Float for precision).
/// Keyed by RetainedViewEntity to match ViewOitItems.
#[derive(Resource, Default)]
pub struct OitTextureCache {
    pub cache: HashMap<bevy::render::view::RetainedViewEntity, OitTextureEntry>,
}

pub struct OitTextureEntry {
    pub texture: Texture,
    pub view: TextureView,
    pub size: (u32, u32),
}

/// Node that clears the OIT accum texture for every view that has OIT content.
/// Ensures we never composite stale content when the camera moves (e.g. if the 3d
/// accum pass is skipped for a view, or view lookup fails).
#[derive(Default)]
pub struct OitClearNode;

impl Node for OitClearNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_oit_items = match world.get_resource::<ViewOitItems>() {
            Some(v) => v,
            None => return Ok(()),
        };
        if view_oit_items.items.is_empty() {
            return Ok(());
        }
        let oit_cache = match world.get_resource::<OitTextureCache>() {
            Some(c) => c,
            None => return Ok(()),
        };
        for (retained_view_entity, lists) in &view_oit_items.items {
            if lists.0.is_empty() && lists.1.is_empty() {
                continue;
            }
            let Some(entry) = oit_cache.cache.get(retained_view_entity) else {
                continue;
            };
            let _pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("oit_clear"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &entry.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations {
                        load: LoadOp::Clear(LinearRgba::new(0.0, 0.0, 0.0, 0.0).into()),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // No draw calls: clear happens on pass start, pass ends when dropped
        }
        Ok(())
    }
}

/// Per-view OIT settings buffers (binding 15 in view bind group). Populated by `prepare_view_oit_settings`.
#[derive(Resource, Default)]
pub struct ViewOitSettingsBuffers {
    pub buffers: HashMap<Entity, Buffer>,
}

/// Default OIT settings buffer used as fallback for bind group binding 15 when a view has no per-view buffer yet.
#[derive(Resource, Default)]
pub struct DefaultOitSettingsBuffer(pub Option<Buffer>);

/// Writes per-view OIT settings to uniform buffers so the view bind group can bind them.
/// Run before `queue_gaussian_view_bind_groups` and `queue_gaussian_compute_view_bind_groups`.
pub fn prepare_view_oit_settings(
    render_device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    mut buffers: ResMut<ViewOitSettingsBuffers>,
    mut default_buffer: ResMut<DefaultOitSettingsBuffer>,
    views: Query<(Entity, Option<&OitSettings>), With<GaussianCamera>>,
) {
    let default_uniform = OitSettingsUniform::default();
    if default_buffer.0.is_none() {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("oit_settings_default_uniform"),
            size: OitSettingsUniform::min_size().get(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&default_uniform));
        default_buffer.0 = Some(buffer);
    }

    for (entity, oit) in &views {
        let s = oit.copied().unwrap_or_default();
        let uniform = OitSettingsUniform {
            depth_weight_scale: s.depth_weight_scale,
            min_weight: s.min_weight,
            depth_weight_power: s.depth_weight_power,
            opacity_scale: s.opacity_scale,
            accum_alpha_scale: s.accum_alpha_scale,
            resolve_opacity_power: s.resolve_opacity_power,
            opacity_bias: s.opacity_bias,
            _pad0: 0.0,
            half_res: s.half_res as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let buffer = buffers.buffers.entry(entity).or_insert_with(|| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some("oit_settings_uniform"),
                size: OitSettingsUniform::min_size().get(),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        // Std140 layout for this struct matches repr(C): f32, f32, vec2 = 16 bytes.
        render_queue.write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
        // #region agent log
        debug_log(
            "H1",
            "src/render/oit.rs:prepare_view_oit_settings",
            "prepared OIT settings buffer",
            serde_json::json!({
                "entity": format!("{entity:?}"),
                "has_oit_component": oit.is_some(),
                "depth_weight_scale": uniform.depth_weight_scale,
                "min_weight": uniform.min_weight
            }),
        );
        // #endregion
    }
    // Remove buffers for despawned views
    buffers.buffers.retain(|e, _| views.get(*e).is_ok());
}

/// Handle for the OIT resolve shader (set when loading the shader).
#[derive(Resource)]
pub struct OitResolveShaderHandle(pub Handle<Shader>);

/// Pipeline and layout for the OIT resolve fullscreen pass.
#[derive(Resource)]
pub struct OitResolvePipeline {
    pub layout: BindGroupLayout,
    pub pipeline_id: CachedRenderPipelineId,
    pub sampler: Sampler,
}

pub struct OitAccumNode<R: PlanarSync> {
    view_query: QueryState<(
        Entity,
        &'static ExtractedView,
        &'static GaussianViewBindGroup,
        &'static ViewUniformOffset,
        Option<Has<MotionVectorPrepass>>,
        Option<&'static PreviousViewUniformOffset>,
        &'static SortTrigger,
        Option<&'static ViewDepthTexture>,
    ), With<GaussianCamera>>,
    cloud_query: QueryState<(
        Entity,
        &'static R::PlanarTypeHandle,
        &'static PlanarStorageBindGroup<R>,
        &'static SortBindGroup,
        &'static DynamicUniformIndex<CloudUniform>,
    )>,
    _phantom: std::marker::PhantomData<R>,
}

impl<R: PlanarSync> OitAccumNode<R> {
    pub fn new(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
            cloud_query: QueryState::new(world),
            _phantom: std::marker::PhantomData,
        }
    }
}


impl<R: PlanarSync> FromWorld for OitAccumNode<R> {
    fn from_world(world: &mut World) -> Self {
        Self::new(world)
    }
}

impl<R: PlanarSync> Node for OitAccumNode<R>
where
    R::GpuPlanarType: GpuPlanarStorage,
{
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
        self.cloud_query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_oit_items = match world.get_resource::<ViewOitItems>() {
            Some(v) => v,
            None => return Ok(()),
        };
        if view_oit_items.items.is_empty() {
            return Ok(());
        }

        let oit_cache = match world.get_resource::<OitTextureCache>() {
            Some(c) => c,
            None => return Ok(()),
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let gaussian_uniforms = world.resource::<GaussianUniformBindGroups>();
        let gaussian_clouds = world.resource::<RenderAssets<R::GpuPlanarType>>();
        let oit_list_idx = if TypeId::of::<R>() == TypeId::of::<Gaussian3d>() {
            0
        } else {
            1
        };
        for (retained_view_entity, lists) in &view_oit_items.items {
            let oit_list = if oit_list_idx == 0 { &lists.0 } else { &lists.1 };
            if oit_list.is_empty() {
                continue;
            }
            let entry = match oit_cache.cache.get(retained_view_entity) {
                Some(e) => e,
                None => continue,
            };
            let mut view_entity_opt = None;
            for (v_entity, ext_view, vg, vo, has_motion, prev_vo, st, depth_tex) in self.view_query.iter_manual(world) {
                if ext_view.retained_view_entity == *retained_view_entity {
                    view_entity_opt = Some((v_entity, vg, vo, has_motion, prev_vo, st, depth_tex));
                    break;
                }
            }
            let (view_bind_group, view_offset, has_motion_vector_prepass, previous_view_offset, sort_trigger, view_depth_texture) =
                match view_entity_opt {
                    Some((_, vg, vo, has_motion, prev_vo, st, depth_tex)) => (vg, vo, has_motion, prev_vo, st, depth_tex),
                    None => continue,
                };
            // View bind group expects 2 dynamic offsets: ViewUniform (binding 0), PreviousViewData (binding 2).
            let previous_offset = match previous_view_offset {
                Some(offset) if has_motion_vector_prepass.unwrap_or_default() => offset.offset,
                _ => 0,
            };
            let view_dynamic_offsets: [u32; 2] = [
                view_offset.offset,
                previous_offset,
            ];

            // Clear is done by OitClearNode; both 3d and 4d use Load so we never show stale content.
            let load_op = LoadOp::Load;

            let depth_view = view_depth_texture
                .map(|depth_tex| depth_tex.texture.create_view(&Default::default()));
            let depth_stencil_attachment = depth_view.as_ref().map(|dv| {
                RenderPassDepthStencilAttachment {
                    view: dv,
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }
            });
            // #region agent log
            debug_log(
                "H3",
                "src/render/oit.rs:OitAccumNode::run",
                "running OIT accum for view",
                serde_json::json!({
                    "retained_view_entity": format!("{retained_view_entity:?}"),
                    "oit_list_len": oit_list.len(),
                    "camera_index": sort_trigger.camera_index,
                    "has_depth_texture": view_depth_texture.is_some(),
                    "accum_size": {"w": entry.size.0, "h": entry.size.1}
                }),
            );
            // #endregion

            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("oit_accum"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &entry.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations {
                        load: load_op,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let Some(base_uniform) = gaussian_uniforms.base_bind_group.as_ref() else {
                continue;
            };

            for (cloud_entity, pipeline_id) in oit_list.iter() {
                let Ok((_, handle, planar_bind_group, sort_bind_group, uniform_index)) =
                    self.cloud_query.get_manual(world, *cloud_entity)
                else {
                    continue;
                };

                let gpu_cloud = match gaussian_clouds.get(handle.handle()) {
                    Some(c) => c,
                    None => continue,
                };

                let Some(pipeline) = pipeline_cache.get_render_pipeline(*pipeline_id) else {
                    continue;
                };

                pass.set_render_pipeline(pipeline);
                pass.set_bind_group(0, &view_bind_group.value, &view_dynamic_offsets);
                pass.set_bind_group(1, base_uniform, &[uniform_index.index()]);
                pass.set_bind_group(2, &planar_bind_group.bind_group, &[]);

                #[cfg(feature = "buffer_storage")]
                {
                    let sort_offset = sort_trigger.camera_index as u32
                        * std::mem::size_of::<SortEntry>() as u32
                        * gpu_cloud.len() as u32;
                    pass.set_bind_group(3, &sort_bind_group.sorted_bind_group, &[sort_offset]);
                }
                #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
                {
                    pass.set_bind_group(3, &sort_bind_group.sorted_bind_group, &[]);
                }

                pass.draw(0..4, 0..gpu_cloud.len() as u32);
            }
        }

        Ok(())
    }
}

/// ViewNode that runs the OIT resolve (sample accum, resolve, composite over scene).
#[derive(Default)]
pub struct OitResolveNode;

impl ViewNode for OitResolveNode {
    type ViewQuery = (
        Entity,
        &'static ExtractedView,
        &'static ViewTarget,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (_view_entity, extracted_view, view_target): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_oit_items = match world.get_resource::<ViewOitItems>() {
            Some(v) => v,
            None => return Ok(()),
        };
        if !view_oit_items.items.contains_key(&extracted_view.retained_view_entity) {
            return Ok(());
        }

        let oit_cache = match world.get_resource::<OitTextureCache>() {
            Some(c) => c,
            None => return Ok(()),
        };
        let entry = match oit_cache.cache.get(&extracted_view.retained_view_entity) {
            Some(e) => e,
            None => return Ok(()),
        };

        let resolve_pipeline = world.resource::<OitResolvePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let view_oit_buffers = world.resource::<ViewOitSettingsBuffers>();
        let default_oit_buffer = world.resource::<DefaultOitSettingsBuffer>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(resolve_pipeline.pipeline_id) else {
            return Ok(());
        };

        let post_process = view_target.post_process_write();

        let oit_resource = view_oit_buffers
            .buffers
            .get(&_view_entity)
            .map(|b| b.as_entire_binding())
            .or_else(|| default_oit_buffer.0.as_ref().map(|b| b.as_entire_binding()))
            .expect("OIT resolve requires view or default OIT settings buffer");

        let bind_group = render_context.render_device().create_bind_group(
            Some("oit_resolve_bind_group"),
            &resolve_pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(post_process.source),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&entry.view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&resolve_pipeline.sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: oit_resource,
                },
            ],
        );

        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("oit_resolve"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                resolve_target: None,
                depth_slice: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);

        Ok(())
    }
}

/// Creates or resizes OIT accum textures for each view that has OIT items.
/// Must run before the OIT accum node (in Render schedule).
pub fn prepare_oit_textures(
    mut oit_cache: ResMut<OitTextureCache>,
    view_oit_items: Res<ViewOitItems>,
    views: Query<
        (
            Entity,
            &ExtractedView,
            &ViewTarget,
            Option<&ViewDepthTexture>,
            Option<&OitSettings>,
        ),
        With<GaussianCamera>,
    >,
    render_device: Res<RenderDevice>,
) {
    oit_cache
        .cache
        .retain(|retained_view_entity, _| view_oit_items.items.contains_key(retained_view_entity));

    for (_view_entity, ext_view, view_target, view_depth_texture, oit_settings) in &views {
        if !view_oit_items.items.contains_key(&ext_view.retained_view_entity) {
            continue;
        }
        let (mut width, mut height) = {
            let tex = view_depth_texture
                .map(|depth| &depth.texture)
                .unwrap_or_else(|| view_target.main_texture());
            (tex.size().width, tex.size().height)
        };
        if oit_settings.map(|s| s.half_res).unwrap_or(false) {
            width = (width / 2).max(1);
            height = (height / 2).max(1);
        }
        // #region agent log
        debug_log(
            "H5",
            "src/render/oit.rs:prepare_oit_textures",
            "prepared OIT target size",
            serde_json::json!({
                "retained_view_entity": format!("{:?}", ext_view.retained_view_entity),
                "width": width,
                "height": height,
                "used_depth_size": view_depth_texture.is_some()
            }),
        );
        // #endregion
        if width == 0 || height == 0 {
            continue;
        }
        let entry = oit_cache.cache.entry(ext_view.retained_view_entity).or_insert_with(|| {
            let desc = TextureDescriptor {
                label: Some("oit_accum"),
                size: Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };
            let texture = render_device.create_texture(&desc);
            let view = texture.create_view(&Default::default());
            OitTextureEntry {
                texture,
                view,
                size: (width, height),
            }
        });
        if entry.size.0 != width || entry.size.1 != height {
            let desc = TextureDescriptor {
                label: Some("oit_accum"),
                size: Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };
            entry.texture = render_device.create_texture(&desc);
            entry.view = entry.texture.create_view(&Default::default());
            entry.size = (width, height);
        }
    }
}

pub fn init_oit_resolve_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: ResMut<PipelineCache>,
    shader_handle: Res<OitResolveShaderHandle>,
) {
    let layout_entries = [
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(OitSettingsUniform::min_size()),
            },
            count: None,
        },
    ];
    let layout_desc = BindGroupLayoutDescriptor::new("oit_resolve_layout", &layout_entries);
    let layout = render_device.create_bind_group_layout(Some("oit_resolve_layout"), &layout_entries);

    let shader = shader_handle.0.clone();
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("oit_resolve".into()),
        layout: vec![layout_desc],
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_fullscreen".into()),
            buffers: vec![],
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("fs_resolve".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        push_constant_ranges: vec![],
        ..default()
    });

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    commands.insert_resource(OitResolvePipeline {
        layout,
        pipeline_id,
        sampler,
    });
}

use crate::render::OIT_RESOLVE_SHADER_HANDLE;

fn clear_view_oit_items(mut view_oit_items: ResMut<ViewOitItems>) {
    view_oit_items.items.clear();
}

/// Registers OIT render graph nodes (accum for 3d/4d + resolve) and resources.
/// Added after RenderPipelinePlugin for both Gaussian3d and Gaussian4d.
pub struct OitRenderGraphPlugin;

impl Plugin for OitRenderGraphPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<OitTextureCache>()
            .init_resource::<ViewOitSettingsBuffers>()
            .init_resource::<DefaultOitSettingsBuffer>()
            .insert_resource(OitResolveShaderHandle(OIT_RESOLVE_SHADER_HANDLE.clone()))
            .add_systems(
                Render,
                clear_view_oit_items
                    .in_set(bevy::render::RenderSystems::Queue)
                    .before(render::queue_gaussians::<Gaussian3d>)
                    .before(render::queue_gaussians::<Gaussian4d>),
            )
            .add_systems(
                Render,
                prepare_view_oit_settings
                    .in_set(bevy::render::RenderSystems::PrepareBindGroups)
                    .before(render::queue_gaussian_view_bind_groups::<Gaussian3d>)
                    .before(render::queue_gaussian_compute_view_bind_groups::<Gaussian3d>),
            )
            .add_systems(
                Render,
                prepare_oit_textures
                    .in_set(bevy::render::RenderSystems::Queue)
                    .after(render::queue_gaussians::<Gaussian3d>)
                    .after(render::queue_gaussians::<Gaussian4d>),
            )
            .add_systems(RenderStartup, init_oit_resolve_pipeline)
            .add_render_graph_node::<OitClearNode>(Core3d, OitClearLabel)
            .add_render_graph_node::<OitAccumNode<Gaussian3d>>(Core3d, OitAccumLabel3d)
            .add_render_graph_node::<OitAccumNode<Gaussian4d>>(Core3d, OitAccumLabel4d)
            .add_render_graph_node::<ViewNodeRunner<OitResolveNode>>(Core3d, OitResolveLabel)
            .add_render_graph_edge(Core3d, Node3d::MainOpaquePass, OitClearLabel)
            .add_render_graph_edge(Core3d, OitClearLabel, OitAccumLabel3d)
            .add_render_graph_edge(Core3d, OitAccumLabel3d, OitAccumLabel4d)
            .add_render_graph_edge(Core3d, OitAccumLabel4d, Node3d::MainTransparentPass)
            .add_render_graph_edge(Core3d, Node3d::MainTransparentPass, OitResolveLabel)
            .add_render_graph_edge(Core3d, OitResolveLabel, Node3d::EndMainPass);
    }
}
