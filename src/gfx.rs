mod camera;
mod model;
mod resources;
mod texture;

use std::sync::Arc;

use glam::{vec3, Quat, Vec3};
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, event::{ElementState, KeyEvent, WindowEvent}, event_loop::EventLoopProxy, keyboard::{KeyCode, PhysicalKey}, window::Window};

use crate::{
    app::WASM_WIN_SIZE,
    gfx::model::Vertex, Instance, InstanceRaw, NUM_INSTANCES_PER_ROW
};

struct CameraState {
    positioning: camera::Camera,
    controller: camera::CameraController,
    uniform: camera::CameraUniform,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}
struct InteractState {
    clear_color: wgpu::Color,
    wireframe: bool,
}
struct ObjectState {
    model: model::Model,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
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

pub struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    render_pipeline: (wgpu::RenderPipeline, Option<wgpu::RenderPipeline>),
    depth_texture: texture::Texture,

    object: ObjectState,
    camera: CameraState,
    interact: InteractState,
}
impl Gfx {
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        // 0 size can cause web-app to crash, better to enforce this than not.
        let size = if cfg!(target_arch = "wasm32") {
            winit::dpi::PhysicalSize {
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
            texture::Texture::create_depth_texture(&device, &surface_config, "depth_texture");

        let diffuse_bytes = include_bytes!("../res/happy-tree.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy_tree.png").unwrap();

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

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let camera = camera::Camera {
            eye: vec3(0., 1., 2.),
            target: Vec3::ZERO,
            up: Vec3::Y,
            aspect: surface_config.width as f32 / surface_config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 1000.0,
        };
        let camera_controller = camera::CameraController::new(0.2);
        let mut camera_uniform = camera::CameraUniform::new();
        camera_uniform.update_view_proj(&camera);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });
        let camera_state = CameraState {
            positioning: camera,
            controller: camera_controller,
            uniform: camera_uniform,
            buffer: camera_buffer,
            bind_group: camera_bind_group,
        };

        const SPACE_BETWEEN: f32 = 3.0;
        let instances = itertools::iproduct!(0..NUM_INSTANCES_PER_ROW, 0..NUM_INSTANCES_PER_ROW)
            .map(|(x, z)| {
                let mapping = |n| SPACE_BETWEEN * (n as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let position = vec3(mapping(x), 0.0, mapping(z));

                // this is needed so an object at (0, 0, 0) won't get scaled to zero
                // as Quaternions can affect scale if they're not created correctly
                let rotation = match position.try_normalize() {
                    Some(position) => Quat::from_axis_angle(position, 45.0),
                    _ => Quat::from_axis_angle(Vec3::Z, 0.0),
                };

                Instance { position, rotation }
            })
            .collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let obj_model =
            resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
                .await
                .unwrap();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = |polygon_mode: wgpu::PolygonMode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: texture::Texture::DEPTH_FORMAT,
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
        };

        let fill_pipeline = render_pipeline(wgpu::PolygonMode::Fill);
        let wireframe_render_pipeline = if (device.features() & wgpu::Features::POLYGON_MODE_LINE)
            == wgpu::Features::POLYGON_MODE_LINE
        {
            Some(render_pipeline(wgpu::PolygonMode::Line))
        } else {
            None
        };

        Self {
            surface,
            device,
            queue,
            surface_config,
            depth_texture,
            camera: camera_state,
            interact: InteractState {
                wireframe: false,
                clear_color: wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                },
            },
            render_pipeline: (fill_pipeline, wireframe_render_pipeline),
            object: ObjectState {
                model: obj_model,
                instances,
                instance_buffer,
                diffuse_bind_group,
                diffuse_texture,
            },
        }
    }

    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        // self.depth_texture =
        // texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub(crate) fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
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
        render_pass.set_pipeline(if self.interact.wireframe {
            self.render_pipeline.1
                .as_ref()
                .unwrap_or(&self.render_pipeline.0)
        } else {
            &self.render_pipeline.0
        });

        use crate::gfx::model::DrawModel;
        render_pass.draw_model_instanced(
            &self.object.model,
            0..self.object.instances.len() as u32,
            &self.camera.bind_group,
        );

        log::debug!("render");

        // drop render pass before we submit to drop the mut borrow on encoder
        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }


    pub fn update(&mut self) {
        self.camera.controller.update_camera(&mut self.camera.positioning);
        self.camera.uniform.update_view_proj(&self.camera.positioning);
        self.queue.write_buffer(
            &self.camera.buffer,
            0,
            bytemuck::cast_slice(&[self.camera.uniform]),
        );
    }

    pub fn input(&mut self, event: &WindowEvent, window_size: PhysicalSize<u32>) -> bool {

        self.camera.controller.process_events(event);

        match event {
            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                self.interact.clear_color = wgpu::Color {
                    r: position.x / (window_size.width as f64),
                    g: 0.2,
                    b: position.y / (window_size.height as f64),
                    a: 0.1,
                };
                return true;
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                if *keycode == KeyCode::KeyL && is_pressed {
                    self.interact.wireframe = !self.interact.wireframe;
                    return true;
                }
            }
            _ => {}
        }

        false
    }
}