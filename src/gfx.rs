mod camera;
mod light;
pub(crate) mod model;
pub(crate) mod primitive;
pub(crate) mod resources;
mod texture;

use crate::gfx::camera::CameraUniform;
use crate::gfx::model::{Material, Mesh, Model, ModelVertex};
use crate::gfx::primitive::cube::Faces;
use crate::gfx::primitive::PrimitiveMeshBuilder;
use crate::gfx::resources::load_model;
use crate::world::chunk::{
    side_lookup, Chunk, ChunkScramble, ChunkTrait, Face, Quad, QuadGroups, CHUNK_SIZE,
};
use crate::world::chunk::{FULL_CHUNK, HALF_CHUNK};
use crate::{
    app::WASM_WIN_SIZE,
    gfx::model::Vertex,
    gui::EguiRenderer,
    world::map::{BlockKind, WorldMap},
    world::World,
    Instance, InstanceRaw,
};
use egui_wgpu::ScreenDescriptor;
use glam::{uvec2, vec3, IVec3, Quat, Vec3};
use itertools::iproduct;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, PI};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, BindingResource, Device};
use winit::{
    dpi::PhysicalSize,
    event::{KeyEvent, WindowEvent},
    event_loop::EventLoopProxy,
    keyboard::PhysicalKey,
    window::Window,
};

pub struct CameraState {
    pub object: camera::Camera,
    pub projection: camera::Projection,
    pub controller: camera::CameraController,
    pub uniform: CameraUniform,
    pub mouse_focused: bool,
}
pub struct InteractState {
    pub clear_color: wgpu::Color,
    pub wireframe: bool,
    pub shadows: bool,
    pub sun_speed: f32,
}
pub struct ObjectState {
    pub models: Vec<Model>,
    pub material: Arc<Vec<Material>>,
    pub instances: Vec<Instance>,
    pub instance_buffer: wgpu::Buffer,
    pub remake: bool,
}

pub struct LightState {
    pub object: camera::Camera,
    pub projection: camera::Projection,
    pub uniform: CameraUniform,
    pub shadow_map: texture::Texture,
}

pub enum MaybeGfx {
    Builder(GfxBuilder),
    Graphics(Gfx),
}

pub struct GfxBuilder {
    event_loop_proxy: Option<EventLoopProxy<Gfx>>,
}

impl GfxBuilder {
    pub fn new(event_loop_proxy: EventLoopProxy<Gfx>) -> Self {
        Self {
            event_loop_proxy: Some(event_loop_proxy),
        }
    }

    pub fn build_and_send(&mut self, window: Arc<Window>) {
        let Some(event_loop_proxy) = self.event_loop_proxy.take() else {
            // event_loop_proxy is already spent - we already constructed Graphics
            return;
        };

        #[cfg(target_arch = "wasm32")]
        {
            //let gfx_fut = create_graphics(event_loop);
            let gfx_fut = Gfx::new(window);
            wasm_bindgen_futures::spawn_local(async move {
                let gfx = gfx_fut.await;
                assert!(event_loop_proxy.send_event(gfx).is_ok());
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let gfx = pollster::block_on(Gfx::new(window));
            assert!(event_loop_proxy.send_event(gfx).is_ok());
        }
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    polygon_mode: wgpu::PolygonMode,
    label: Option<&str>,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label,
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: vertex_layouts,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // cull_mode: None,
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

pub enum OneOrTwo<T> {
    One(T),
    Two(T, T),
}

impl<T> OneOrTwo<T> {
    pub fn one(&mut self) -> &mut T {
        match self {
            OneOrTwo::One(t) => t,
            OneOrTwo::Two(_, _) => {
                panic!("Used one() on a Two.")
            }
        }
    }
    pub fn two(&mut self) -> (&mut T, &mut T) {
        match self {
            OneOrTwo::One(_) => panic!("Used two() on a One."),
            OneOrTwo::Two(x, y) => (x, y),
        }
    }
}
pub struct Pass {
    pipeline: OneOrTwo<wgpu::RenderPipeline>,
    bind_group: wgpu::BindGroup,
    uniform_bufs: Vec<wgpu::Buffer>,
}

pub struct Gfx {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    // pub render_pipelines: RenderPipelines,
    pub depth_texture: texture::Texture,
    pub texture_bind_group_layout: BindGroupLayout,

    // pub shadow_map_buffer: wgpu::Buffer,
    // pub shadow_bind_group: wgpu::BindGroup,
    pub shadow_pass: Pass,
    pub forward_pass: Pass,

    pub object: ObjectState,
    pub camera: CameraState,
    pub interact: InteractState,
    pub light: LightState,
}
impl Gfx {
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        // 0 size can cause web-app to crash, better to enforce this than not.
        let size = if cfg!(target_arch = "wasm32") {
            PhysicalSize {
                width: WASM_WIN_SIZE.0,
                height: WASM_WIN_SIZE.1,
            }
        } else {
            window.inner_size()
        };

        // Handle to our GPU backends
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,

            // WebGPU is sadly not supported in most browsers (yet!)
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,

            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                // lo-power or hi-perf
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: if adapter.get_info().backend == wgpu::Backend::Gl {
                        wgpu::Features::empty()
                    } else {
                        wgpu::Features::POLYGON_MODE_LINE
                    },
                    // required_features: wgpu::Features::empty(),

                    // WebGL does not support all of wgpu's features
                    required_limits: if adapter.get_info().backend == wgpu::Backend::Gl {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        // sRGB surface textures assumed
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0], // eg vsync
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &surface_config, None, "depth_texture");

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let camera = camera::Camera::new(vec3(10., 10., -5.), 2.4, -1.);
        let projection = camera::Projection::new(
            uvec2(surface_config.width, surface_config.height).as_vec2(),
            FRAC_PI_2,
            0.1,
            1000.0,
        );
        let camera_controller = camera::CameraController::new(40.0, 0.4);

        let mut camera_uniform = camera::CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_state = CameraState {
            object: camera,
            projection,
            controller: camera_controller,
            uniform: camera_uniform,
            mouse_focused: false,
        };

        // let test_chunk = QuadGroups {
        //     groups: [
        //         vec![Quad {
        //             voxel: [1, 1, 0],
        //             width: 2,
        //             height: 3,
        //         }],
        //         vec![Quad {
        //             voxel: [1, 1, 1],
        //             width: 2,
        //             height: 3,
        //         }],
        //         vec![Quad {
        //             voxel: [0, 0, 0],
        //             width: 2,
        //             height: 9,
        //         }],
        //         vec![Quad {
        //             voxel: [0, 0, 1],
        //             width: 2,
        //             height: 9,
        //         }],
        //         vec![Quad {
        //             voxel: [0, 1, 1],
        //             width: 2,
        //             height: 9,
        //         }],
        //         vec![Quad {
        //             voxel: [0, 1, 0],
        //             width: 2,
        //             height: 9,
        //         }],
        //     ],
        // };
        //
        // let model = Model {
        //     meshes: vec![model],
        //     materials: resources::load_model(
        //         "blender_default_cube.obj",
        //         &device,
        //         &queue,
        //         &texture_bind_group_layout,
        //     )
        //     .await
        //     .unwrap()
        //     .materials,
        // };

        let mut models = vec![];
        // models.push(
        //     resources::load_model(
        //         "blender_default_cube.obj",
        //         &device,
        //         &queue,
        //         &texture_bind_group_layout,
        //     )
        //     .await
        //     .unwrap(),
        // );
        // models.push(
        //     resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
        //         .await
        //         .unwrap(),
        // );
        // models.push(
        //     primitive::PrimitiveMesh::new(
        //         &device,
        //         &queue,
        //         &texture_bind_group_layout,
        //         primitive::cube::cube_vertices(1.0).as_ref(),
        //         primitive::cube::cube_indices().as_ref(), /* &std::vec::Vec<u32> */
        //     )
        //     .await
        //     .model,
        // );
        // models.push({
        //     PrimitiveMeshBuilder::new()
        //         // .cube(0., 0., 0.)
        //         .cube(Faces::ALL, 3., 0., 0.)
        //         .cube(Faces::ALL, -3., 0., 0.)
        //         .build(&device, &queue, &texture_bind_group_layout)
        //         .await
        //         .model
        // });

        // models.push(
        //     HALF_CHUNK()
        //         .model(&device, &queue, &texture_bind_group_layout)
        //         .await
        //         .unwrap(),
        // );
        // models.push({
        //     let mut buffer = QuadGroups::default();
        //
        //     let block_pos = [0, 2, 0];
        //     let mut quads = (0..6)
        //         .map(side_lookup)
        //         .map(|i| Quad {
        //             voxel: block_pos,
        //             width: 2,
        //             height: 2,
        //         })
        //         .collect_vec();
        //
        //     buffer.groups[0].append(&mut quads);
        //
        //     let mesh = buffer.mesh(&device, 0);
        //     Model {
        //         meshes: resources::load_model(
        //             "blender_default_cube.obj",
        //             &device,
        //             &queue,
        //             &texture_bind_group_layout,
        //         )
        //         .await
        //         .unwrap()
        //         .meshes,
        //         materials: resources::load_model(
        //             "cube.obj",
        //             &device,
        //             &queue,
        //             &texture_bind_group_layout,
        //         )
        //         .await
        //         .unwrap()
        //         .materials,
        //     }
        // });

        let material = load_model(
            "blender_default_cube.obj",
            &device,
            &queue,
            &texture_bind_group_layout,
        )
        .await
        .unwrap()
        .materials;

        // MAP LOAD
        let map = crate::world::map::new();
        Self::remake_mesh(&map, &mut models, &material, &device);

        let instances = vec![
            (Instance {
                position: Default::default(),
                rotation: Default::default(),
            }),
        ];

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        log::info!("Light setup!");
        let light = camera::Camera::new(vec3(0., 300., 0.), -90.0, -20.0);
        let light_projection = camera::Projection::new(
            uvec2(surface_config.width, surface_config.height).as_vec2(),
            FRAC_PI_2,
            0.1,
            1000.0,
        );

        let mut light_uniform = camera::CameraUniform::new();
        light_uniform.update_view_proj(&light, &light_projection);

        let shadow_map = texture::Texture::create_depth_texture(
            &device,
            &surface_config,
            Some({
                if cfg!(target_arch = "wasm32") {
                    (2048, 2048)
                } else {
                    (5000, 5000)
                }
            }),
            "Shadow Map",
        );

        fn bgl_t(binding: u32) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        }
        fn bge(binding: u32, resource: BindingResource<'_>) -> wgpu::BindGroupEntry {
            wgpu::BindGroupEntry { binding, resource }
        }

        log::info!("Pipeline/Shadow setup!");

        let shadow_pass = {
            let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Camera Buffer"),
                size: size_of::<camera::CameraUniform>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Light VB"),
                size: size_of::<CameraUniform>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        // Camera
                        bgl_t(0),
                        // Light position
                        bgl_t(1),
                    ],
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    bge(0, camera_buffer.as_entire_binding()),
                    bge(1, light_buffer.as_entire_binding()),
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shadow Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shadows.wgsl").into()),
            };

            let shader = device.create_shader_module(shader);

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Shadows Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
                    compilation_options: Default::default(),
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Front),
                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                    polygon_mode: wgpu::PolygonMode::Fill,
                    // Requires Features::DEPTH_CLIP_CONTROL
                    unclipped_depth: false,
                    // Requires Features::CONSERVATIVE_RASTERIZATION
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

            Pass {
                pipeline: OneOrTwo::One(pipeline),
                bind_group,
                uniform_bufs: vec![camera_buffer, light_buffer],
            }
        };

        let forward_pass = {
            let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
            let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Light VB"),
                contents: bytemuck::cast_slice(&[light_uniform]),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        // Camera positional Data
                        bgl_t(0),
                        // Light positional data
                        bgl_t(1),
                        // Shadowmap
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                            count: None,
                        },
                    ],
                    label: None,
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    bge(0, camera_buffer.as_entire_binding()),
                    bge(1, light_buffer.as_entire_binding()),
                    bge(2, BindingResource::TextureView(&shadow_map.view)),
                    bge(3, BindingResource::Sampler(&shadow_map.sampler)),
                ],
                label: None,
            });

            let render_pipeline = create_render_pipeline(
                &device,
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&texture_bind_group_layout, &bind_group_layout],
                    push_constant_ranges: &[],
                }),
                surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ModelVertex::desc(), InstanceRaw::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Normal Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
                },
                wgpu::PolygonMode::Fill, // wireframe line wireframe
                Some("Normal Render Pipeline"),
            );

            log::info!("Pipeline/Light setup!");
            let light_render_pipeline = create_render_pipeline(
                &device,
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Light Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                }),
                surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Light Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
                },
                wgpu::PolygonMode::Fill,
                Some("Light Render Pipeline"),
            );

            Pass {
                pipeline: OneOrTwo::Two(render_pipeline, light_render_pipeline),
                bind_group,
                uniform_bufs: vec![camera_buffer, light_buffer],
            }
        };

        log::debug!("Initialized!");

        Self {
            surface,
            device,
            queue,
            surface_config,
            depth_texture,
            camera: camera_state,
            shadow_pass,
            forward_pass,
            texture_bind_group_layout,
            interact: InteractState {
                wireframe: false,
                clear_color: wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                },
                shadows: true,
                sun_speed: FRAC_PI_3,
            },
            object: ObjectState {
                models,
                instances,
                instance_buffer,
                remake: false,
                material,
            },
            light: LightState {
                object: light,
                projection: light_projection,
                uniform: light_uniform,
                shadow_map,
            },
        }
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.camera
            .projection
            .resize(uvec2(self.surface_config.width, self.surface_config.height).as_vec2());

        self.surface.configure(&self.device, &self.surface_config);

        self.depth_texture = texture::Texture::create_depth_texture(
            &self.device,
            &self.surface_config,
            None,
            "depth_texture",
        );
    }

    fn remake_mesh(
        map: &WorldMap,
        models: &mut Vec<Model>,
        mats: &Arc<Vec<Material>>,
        device: &Device,
    ) {
        models.clear();
        for (chunk_pos, chunk) in map.chunks.iter() {
            if let Some(mesh) = chunk.primitive_model(chunk_pos, device, mats) {
                models.push(mesh);
            }
        }
    }

    pub(crate) fn render(
        &mut self,
        egui: &mut Option<EguiRenderer>,
        window: Arc<Window>,
        world: &mut World,
        dt: instant::Duration,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        encoder.push_debug_group("Shadow pass");
        use crate::gfx::model::DrawModel;

        // Shadow-map Pass
        if self.interact.shadows {
            let Pass {
                pipeline,
                bind_group: _,
                uniform_bufs,
            } = &mut self.shadow_pass;

            encoder.copy_buffer_to_buffer(
                &self.forward_pass.uniform_bufs[0],
                0,
                &uniform_bufs[0],
                0,
                size_of::<CameraUniform>() as wgpu::BufferAddress,
            );
            encoder.copy_buffer_to_buffer(
                &self.forward_pass.uniform_bufs[1],
                0,
                &uniform_bufs[1],
                0,
                size_of::<CameraUniform>() as wgpu::BufferAddress,
            );

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Render Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.light.shadow_map.view, // Shadow map written here
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(pipeline.one());
            render_pass.set_vertex_buffer(1, self.object.instance_buffer.slice(..));

            // render_pass.draw_light_model_instanced(
            //     &self.object.model,
            //     0..self.object.instances.len() as u32,
            //     &[bind_group],
            // );
        }

        encoder.pop_debug_group();

        encoder.push_debug_group("Forward pass");
        {
            let Pass {
                pipeline,
                bind_group,
                uniform_bufs: _,
            } = &mut self.forward_pass;

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Forward Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.interact.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_vertex_buffer(1, self.object.instance_buffer.slice(..));

            use crate::gfx::model::DrawLight;

            render_pass.set_pipeline(pipeline.two().1);
            // render_pass.draw_light_model(&self.object.model, &[bind_group]);

            render_pass.set_pipeline(pipeline.two().0);
            for model in &self.object.models {
                render_pass.draw_model(model, &[bind_group]);
            }

            // render_pass.draw_mesh(
            //     &self.object.world_mesh,
            //     &self.object.model.materials[0],
            //     &[bind_group],
            // );
            // let mesh = &self.object.world_mesh;
            // render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..784));
            // render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            // render_pass.set_bind_group(0, &self.object.model.materials[0].bind_group, &[]);
            // render_pass.set_bind_group(1, Some(&*bind_group), &[]);
            // render_pass.draw(0..mesh.num_elements, 0..1);
        }

        encoder.pop_debug_group();

        // Layer EGUI on top of frame!

        if let Some(egui) = egui {
            let pixels_per_point = if cfg!(target_arch = "wasm32") {
                1.0
            } else {
                window.scale_factor() as f32 * egui.scale_factor
            };
            let screen_descriptor = ScreenDescriptor {
                size_in_pixels: [self.surface_config.width, self.surface_config.height],
                pixels_per_point,
            };
            egui.begin_frame(&window);

            egui.update(self, world, dt);

            egui.end_frame_and_draw(
                &self.device,
                &self.queue,
                &mut encoder,
                &window,
                &view,
                screen_descriptor,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn update(&mut self, world: &mut World, dt: instant::Duration) {
        // Camera update
        self.camera.controller.update_camera(
            &mut self.camera.object,
            dt,
            world,
            &mut self.object.remake,
        );
        self.camera
            .uniform
            .update_view_proj(&self.camera.object, &self.camera.projection);

        self.queue.write_buffer(
            &self.forward_pass.uniform_bufs[0],
            0,
            bytemuck::cast_slice(&[self.camera.uniform]),
        );

        // Light update
        self.light.object.position = Quat::from_axis_angle(
            vec3(0.0, 0.0, 1.0),
            self.interact.sun_speed * dt.as_secs_f32(),
        ) * self.light.object.position;
        self.light
            .uniform
            .update_view_proj(&self.light.object, &self.light.projection);

        self.queue.write_buffer(
            &self.forward_pass.uniform_bufs[1],
            0,
            bytemuck::cast_slice(&[self.light.uniform]),
        );

        // Object update
        if self.object.remake {
            Self::remake_mesh(
                &world.map,
                &mut self.object.models,
                &self.object.material,
                &self.device,
            );
            self.object.remake = false;
        }
    }

    pub fn input(&mut self, event: &WindowEvent, _window_size: PhysicalSize<u32>) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => self.camera.controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera.controller.process_scroll(delta);
                true
            }
            _ => false,
        }
    }
}
