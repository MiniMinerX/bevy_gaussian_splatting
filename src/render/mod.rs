use std::hash::Hash;

use bevy::{
    prelude::*,
    asset::{
        load_internal_asset,
        LoadState,
    },
    core_pipeline::core_3d::Transparent3d,
    ecs::{
        system::{
            lifetimeless::*,
            SystemParamItem,
        },
        query::ROQueryItem,
    },
    render::{
        Extract,
        extract_component::{
            DynamicUniformIndex,
            UniformComponentPlugin,
            ComponentUniforms,
        },
        globals::{
            GlobalsUniform,
            GlobalsBuffer,
        },
        render_asset::{
            PrepareAssetError,
            RenderAsset,
            RenderAssets,
            RenderAssetPlugin,
        },
        render_phase::{
            AddRenderCommand,
            DrawFunctions,
            PhaseItem,
            RenderCommand,
            RenderCommandResult,
            RenderPhase,
            SetItemPipeline,
            TrackedRenderPass,
        },
        render_resource::*,
        renderer::RenderDevice,
        Render,
        RenderApp,
        RenderSet,
        view::{
            ExtractedView,
            ViewUniform,
            ViewUniforms,
            ViewUniformOffset,
        },
    },
};

use crate::{
    gaussian::{
        Gaussian,
        GaussianCloud,
        GaussianCloudSettings,
        MAX_SH_COEFF_COUNT,
    },
    render::{
        morph::MorphPlugin,
        sort::RadixSortPlugin,
    },
};

use self::sort::GpuRadixBuffers;

pub mod morph;
pub mod sort;


const BINDINGS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(675257236);
const GAUSSIAN_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(68294581);
const SPHERICAL_HARMONICS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(834667312);
const TRANSFORM_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(734523534);


#[derive(Default)]
pub struct RenderPipelinePlugin;

impl Plugin for RenderPipelinePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            BINDINGS_SHADER_HANDLE,
            "bindings.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            GAUSSIAN_SHADER_HANDLE,
            "gaussian.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            SPHERICAL_HARMONICS_SHADER_HANDLE,
            "spherical_harmonics.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            TRANSFORM_SHADER_HANDLE,
            "transform.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(RenderAssetPlugin::<GaussianCloud>::default());
        app.add_plugins(UniformComponentPlugin::<GaussianCloudUniform>::default());
        app.add_plugins((
            MorphPlugin,
            RadixSortPlugin,
        ));

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Transparent3d, DrawGaussians>()
                .init_resource::<GaussianUniformBindGroups>()
                .add_systems(ExtractSchedule, extract_gaussians)
                .add_systems(
                    Render,
                    (
                        queue_gaussian_bind_group.in_set(RenderSet::QueueMeshes),
                        queue_gaussian_view_bind_groups.in_set(RenderSet::QueueMeshes),
                        queue_gaussians.in_set(RenderSet::QueueMeshes),
                    ),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<GaussianCloudPipeline>()
                .init_resource::<SpecializedRenderPipelines<GaussianCloudPipeline>>();
        }
    }
}


#[derive(Bundle)]
pub struct GpuGaussianSplattingBundle {
    pub settings: GaussianCloudSettings,
    pub settings_uniform: GaussianCloudUniform,
    pub verticies: Handle<GaussianCloud>,
}

#[derive(Debug, Clone)]
pub struct GpuGaussianCloud {
    pub gaussian_buffer: Buffer,
    pub count: usize,

    pub draw_indirect_buffer: Buffer,

    pub radix_sort_buffers: GpuRadixBuffers,
}
impl RenderAsset for GaussianCloud {
    type ExtractedAsset = GaussianCloud;
    type PreparedAsset = GpuGaussianCloud;
    type Param = SRes<RenderDevice>;

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        gaussian_cloud: Self::ExtractedAsset,
        render_device: &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let gaussian_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian cloud buffer"),
            contents: bytemuck::cast_slice(gaussian_cloud.gaussians.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        });

        let count = gaussian_cloud.gaussians.len();

        let draw_indirect_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("draw indirect buffer"),
            size: std::mem::size_of::<wgpu::util::DrawIndirect>() as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok(GpuGaussianCloud {
            gaussian_buffer,
            count,
            draw_indirect_buffer,
            radix_sort_buffers: GpuRadixBuffers::new(count, render_device),
        })
    }
}


#[allow(clippy::too_many_arguments)]
fn queue_gaussians(
    gaussian_cloud_uniform: Res<ComponentUniforms<GaussianCloudUniform>>,
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<GaussianCloudPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<GaussianCloudPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    gaussian_clouds: Res<RenderAssets<GaussianCloud>>,
    gaussian_splatting_bundles: Query<(
        Entity,
        &Handle<GaussianCloud>,
        &GaussianCloudSettings,
    )>,
    mut views: Query<(
        &ExtractedView,
        &mut RenderPhase<Transparent3d>,
    )>,
) {
    // TODO: condition this system based on GaussianCloudBindGroup attachment
    if gaussian_cloud_uniform.buffer().is_none() {
        return;
    };

    let draw_custom = transparent_3d_draw_functions.read().id::<DrawGaussians>();

    for (_view, mut transparent_phase) in &mut views {
        for (entity, cloud, settings) in &gaussian_splatting_bundles {
            if let Some(_cloud) = gaussian_clouds.get(cloud) {
                let key = GaussianCloudPipelineKey {
                    aabb: settings.aabb,
                    visualize_bounding_box: settings.visualize_bounding_box,
                };

                let pipeline = pipelines.specialize(&pipeline_cache, &custom_pipeline, key);

                // // TODO: distance to gaussian cloud centroid
                // let rangefinder = view.rangefinder3d();

                transparent_phase.add(Transparent3d {
                    entity,
                    draw_function: draw_custom,
                    distance: 0.0,
                    // distance: rangefinder
                    //     .distance_translation(&mesh_instance.transforms.transform.translation),
                    pipeline,
                    batch_range: 0..1,
                    dynamic_offset: None,
                });
            }
        }
    }
}




#[derive(Resource)]
pub struct GaussianCloudPipeline {
    shader: Handle<Shader>,
    pub gaussian_cloud_layout: BindGroupLayout,
    pub gaussian_uniform_layout: BindGroupLayout,
    pub view_layout: BindGroupLayout,
    pub sorted_layout: BindGroupLayout,
}

impl FromWorld for GaussianCloudPipeline {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();

        let view_layout_entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(GlobalsUniform::min_size()),
                },
                count: None,
            },
        ];

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_view_layout"),
            entries: &view_layout_entries,
        });

        let gaussian_uniform_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_uniform_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(GaussianCloudUniform::min_size()),
                    },
                    count: None,
                },
            ],
        });

        let gaussian_cloud_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_cloud_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(std::mem::size_of::<Gaussian>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let sorted_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sorted_layout"),
            entries: &vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(std::mem::size_of::<(u32, u32)>() as u64),
                    },
                    count: None,
                },
            ],
        });

        GaussianCloudPipeline {
            gaussian_cloud_layout,
            gaussian_uniform_layout,
            view_layout,
            shader: GAUSSIAN_SHADER_HANDLE,
            sorted_layout,
        }
    }
}

// TODO: allow setting shader defines via API
// TODO: separate shader defines for each pipeline
struct ShaderDefines {
    radix_bits_per_digit: u32,
    radix_digit_places: u32,
    radix_base: u32,
    entries_per_invocation_a: u32,
    entries_per_invocation_c: u32,
    workgroup_invocations_a: u32,
    workgroup_invocations_c: u32,
    workgroup_entries_a: u32,
    workgroup_entries_c: u32,
    sorting_buffer_size: u32,

    temporal_sort_window_size: u32,
}

impl ShaderDefines {
    fn max_tile_count(&self, count: usize) -> u32 {
        (count as u32 + self.workgroup_entries_c - 1) / self.workgroup_entries_c
    }

    fn sorting_status_counters_buffer_size(&self, count: usize) -> usize {
        self.radix_base as usize * self.max_tile_count(count) as usize * std::mem::size_of::<u32>()
    }
}

impl Default for ShaderDefines {
    fn default() -> Self {
        let radix_bits_per_digit = 8;
        let radix_digit_places = 32 / radix_bits_per_digit;
        let radix_base = 1 << radix_bits_per_digit;
        let entries_per_invocation_a = 4;
        let entries_per_invocation_c = 4;
        let workgroup_invocations_a = radix_base * radix_digit_places;
        let workgroup_invocations_c = radix_base;
        let workgroup_entries_a = workgroup_invocations_a * entries_per_invocation_a;
        let workgroup_entries_c = workgroup_invocations_c * entries_per_invocation_c;
        let sorting_buffer_size = radix_base * radix_digit_places *
            std::mem::size_of::<u32>() as u32 + 5 * std::mem::size_of::<u32>() as u32;

        Self {
            radix_bits_per_digit,
            radix_digit_places,
            radix_base,
            entries_per_invocation_a,
            entries_per_invocation_c,
            workgroup_invocations_a,
            workgroup_invocations_c,
            workgroup_entries_a,
            workgroup_entries_c,
            sorting_buffer_size,

            temporal_sort_window_size: 16,
        }
    }
}

fn shader_defs(
    aabb: bool,
    visualize_bounding_box: bool,
) -> Vec<ShaderDefVal> {
    let defines = ShaderDefines::default();
    let mut shader_defs = vec![
        ShaderDefVal::UInt("MAX_SH_COEFF_COUNT".into(), MAX_SH_COEFF_COUNT as u32),
        ShaderDefVal::UInt("RADIX_BASE".into(), defines.radix_base),
        ShaderDefVal::UInt("RADIX_BITS_PER_DIGIT".into(), defines.radix_bits_per_digit),
        ShaderDefVal::UInt("RADIX_DIGIT_PLACES".into(), defines.radix_digit_places),
        ShaderDefVal::UInt("ENTRIES_PER_INVOCATION_A".into(), defines.entries_per_invocation_a),
        ShaderDefVal::UInt("ENTRIES_PER_INVOCATION_C".into(), defines.entries_per_invocation_c),
        ShaderDefVal::UInt("WORKGROUP_INVOCATIONS_A".into(), defines.workgroup_invocations_a),
        ShaderDefVal::UInt("WORKGROUP_INVOCATIONS_C".into(), defines.workgroup_invocations_c),
        ShaderDefVal::UInt("WORKGROUP_ENTRIES_C".into(), defines.workgroup_entries_c),

        ShaderDefVal::UInt("TEMPORAL_SORT_WINDOW_SIZE".into(), defines.temporal_sort_window_size),
    ];

    if aabb {
        shader_defs.push("USE_AABB".into());
    }

    if !aabb {
        shader_defs.push("USE_OBB".into());
    }

    if visualize_bounding_box {
        shader_defs.push("VISUALIZE_BOUNDING_BOX".into());
    }

    shader_defs
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct GaussianCloudPipelineKey {
    pub aabb: bool,
    pub visualize_bounding_box: bool,
}

impl SpecializedRenderPipeline for GaussianCloudPipeline {
    type Key = GaussianCloudPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let shader_defs = shader_defs(
            key.aabb,
            key.visualize_bounding_box,
        );

        RenderPipelineDescriptor {
            label: Some("gaussian cloud render pipeline".into()),
            layout: vec![
                self.view_layout.clone(),
                self.gaussian_uniform_layout.clone(),
                self.gaussian_cloud_layout.clone(),
                self.sorted_layout.clone(),
            ],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                entry_point: "vs_points".into(),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: "fs_main".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::DstAlpha,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                unclipped_depth: false,
                cull_mode: None,
                conservative: false,
                polygon_mode: PolygonMode::Fill,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: 4,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            push_constant_ranges: Vec::new(),
        }
    }
}

type DrawGaussians = (
    SetItemPipeline,
    SetGaussianViewBindGroup<0>,
    SetGaussianUniformBindGroup<1>,
    DrawGaussianInstanced,
);


#[derive(Component, ShaderType, Clone)]
pub struct GaussianCloudUniform {
    pub transform: Mat4,
    pub global_scale: f32,
}

pub fn extract_gaussians(
    mut commands: Commands,
    mut prev_commands_len: Local<usize>,
    gaussians_query: Extract<
        Query<(
            Entity,
            // &ComputedVisibility,
            &Visibility,
            &Handle<GaussianCloud>,
            &GaussianCloudSettings,
        )>,
    >,
) {
    let mut commands_list = Vec::with_capacity(*prev_commands_len);
    // let visible_gaussians = gaussians_query.iter().filter(|(_, vis, ..)| vis.is_visible());

    for (
        entity,
        visibility,
        verticies,
        settings,
    ) in gaussians_query.iter() {
        if visibility == Visibility::Hidden {
            continue;
        }

        let settings_uniform = GaussianCloudUniform {
            transform: settings.global_transform.compute_matrix(),
            global_scale: settings.global_scale,
        };
        commands_list.push((
            entity,
            GpuGaussianSplattingBundle {
                settings: settings.clone(),
                settings_uniform,
                verticies: verticies.clone(),
            },
        ));
    }
    *prev_commands_len = commands_list.len();
    commands.insert_or_spawn_batch(commands_list);
}


#[derive(Resource, Default)]
pub struct GaussianUniformBindGroups {
    base_bind_group: Option<BindGroup>,
}

#[derive(Component)]
pub struct GaussianCloudBindGroup {
    pub cloud_bind_group: BindGroup,
    pub sorted_bind_group: BindGroup,
}

fn queue_gaussian_bind_group(
    mut commands: Commands,
    mut groups: ResMut<GaussianUniformBindGroups>,
    gaussian_cloud_pipeline: Res<GaussianCloudPipeline>,
    render_device: Res<RenderDevice>,
    gaussian_uniforms: Res<ComponentUniforms<GaussianCloudUniform>>,
    asset_server: Res<AssetServer>,
    gaussian_cloud_res: Res<RenderAssets<GaussianCloud>>,
    gaussian_clouds: Query<(
        Entity,
        &Handle<GaussianCloud>,
    )>,
) {
    let Some(model) = gaussian_uniforms.buffer() else {
        return;
    };

    groups.base_bind_group = Some(render_device.create_bind_group(
        "gaussian_uniform_bind_group",
        &gaussian_cloud_pipeline.gaussian_uniform_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: model,
                    offset: 0,
                    size: GaussianCloudUniform::min_size().into(),
                }),
            },
        ],
    ));

    for (entity, cloud_handle) in gaussian_clouds.iter() {
        // TODO: add asset loading indicator (and maybe streamed loading)
        if Some(LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        if gaussian_cloud_res.get(cloud_handle).is_none() {
            continue;
        }

        let cloud = gaussian_cloud_res.get(cloud_handle).unwrap();

        commands.entity(entity).insert(GaussianCloudBindGroup {
            cloud_bind_group: render_device.create_bind_group(
                "gaussian_cloud_bind_group",
                &gaussian_cloud_pipeline.gaussian_cloud_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &cloud.gaussian_buffer,
                            offset: 0,
                            size: BufferSize::new(cloud.gaussian_buffer.size()),
                        }),
                    },
                ],
            ),
            sorted_bind_group: render_device.create_bind_group(
                "render_sorted_bind_group",
                &gaussian_cloud_pipeline.sorted_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &cloud.radix_sort_buffers.entry_buffer_a,
                            offset: 0,
                            size: BufferSize::new((cloud.count as usize * std::mem::size_of::<(u32, u32)>()) as u64),
                        }),
                    },
                ],
            ),
        });
    }
}

#[derive(Component)]
pub struct GaussianViewBindGroup {
    pub value: BindGroup,
}

pub fn queue_gaussian_view_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    gaussian_cloud_pipeline: Res<GaussianCloudPipeline>,
    view_uniforms: Res<ViewUniforms>,
    views: Query<(
        Entity,
        &ExtractedView,
        &mut RenderPhase<Transparent3d>,
    )>,
    globals_buffer: Res<GlobalsBuffer>,
) {
    if let (
        Some(view_binding),
        Some(globals),
    ) = (
        view_uniforms.uniforms.binding(),
        globals_buffer.buffer.binding(),
    ) {
        for (
            entity,
            _extracted_view,
            _render_phase,
        ) in &views
        {
            let layout = &gaussian_cloud_pipeline.view_layout;

            let entries = vec![
                BindGroupEntry {
                    binding: 0,
                    resource: view_binding.clone(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: globals.clone(),
                },
            ];

            let view_bind_group = render_device.create_bind_group(
                "gaussian_view_bind_group",
                layout,
                &entries,
            );

            commands.entity(entity).insert(GaussianViewBindGroup {
                value: view_bind_group,
            });
        }
    }
}

pub struct SetGaussianViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetGaussianViewBindGroup<I> {
    type Param = ();
    type ViewWorldQuery = (
        Read<ViewUniformOffset>,
        Read<GaussianViewBindGroup>,
    );
    type ItemWorldQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        (view_uniform, gaussian_view_bind_group): ROQueryItem<
            'w,
            Self::ViewWorldQuery,
        >,
        _entity: (),
        _: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(
            I,
            &gaussian_view_bind_group.value,
            &[view_uniform.offset],
        );

        RenderCommandResult::Success
    }
}


pub struct SetGaussianUniformBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetGaussianUniformBindGroup<I> {
    type Param = SRes<GaussianUniformBindGroups>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<DynamicUniformIndex<GaussianCloudUniform>>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        gaussian_cloud_index: ROQueryItem<Self::ItemWorldQuery>,
        bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let bind_groups = bind_groups.into_inner();
        let bind_group = bind_groups.base_bind_group.as_ref().expect("bind group not initialized");

        let mut set_bind_group = |indices: &[u32]| pass.set_bind_group(I, bind_group, indices);
        let gaussian_cloud_index = gaussian_cloud_index.index();
        set_bind_group(&[gaussian_cloud_index]);

        RenderCommandResult::Success
    }
}

pub struct DrawGaussianInstanced;
impl<P: PhaseItem> RenderCommand<P> for DrawGaussianInstanced {
    type Param = SRes<RenderAssets<GaussianCloud>>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = (
        Read<Handle<GaussianCloud>>,
        Read<GaussianCloudBindGroup>,
    );

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        (
            handle,
            bind_groups,
        ): (
            &'w Handle<GaussianCloud>,
            &'w GaussianCloudBindGroup,
        ),
        gaussian_clouds: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_gaussian_cloud = match gaussian_clouds.into_inner().get(handle) {
            Some(gpu_gaussian_cloud) => gpu_gaussian_cloud,
            None => return RenderCommandResult::Failure,
        };

        pass.set_bind_group(2, &bind_groups.cloud_bind_group, &[]);
        pass.set_bind_group(3, &bind_groups.sorted_bind_group, &[]);

        pass.draw_indirect(&gpu_gaussian_cloud.draw_indirect_buffer, 0);

        RenderCommandResult::Success
    }
}
