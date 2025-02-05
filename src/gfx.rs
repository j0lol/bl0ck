mod camera;
mod model;
mod resources;
mod texture;

use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use glam::{vec3, Quat, Vec3};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::EventLoopProxy,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::{
    app::WASM_WIN_SIZE,
    gfx::model::Vertex,
    gui::EguiRenderer,
    map::{sl3get, Block, WorldMap, CHUNK_SIZE},
    Instance, InstanceRaw,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: Vec3,
    _pad: u32,
    color: Vec3,
    _pad2: u32,
}
impl LightUniform {
    pub fn new(position: Vec3, color: Vec3) -> LightUniform {
        LightUniform {
            position,
            color,
            _pad: 0,
            _pad2: 0,
        }
    }
}

pub struct CameraState {
    pub positioning: camera::Camera,
    pub controller: camera::CameraController,
    pub uniform: camera::CameraUniform,
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}
pub struct InteractState {
    pub clear_color: wgpu::Color,
    pub wireframe: bool,
}
pub struct ObjectState {
    pub model: model::Model,
    pub instances: Vec<Instance>,
    pub instance_buffer: wgpu::Buffer,
    pub remake: bool,
}

pub struct LightState {
    pub uniform: LightUniform,
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

pub struct RenderPipelines {
    camera: wgpu::RenderPipeline,
    camera_wireframe: Option<wgpu::RenderPipeline>,
    light: wgpu::RenderPipeline,
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
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
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

pub struct Gfx {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub render_pipelines: RenderPipelines,
    pub depth_texture: texture::Texture,

    pub map: WorldMap,
    pub object: ObjectState,
    pub camera: CameraState,
    pub interact: InteractState,
    pub light: LightState,
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

        let camera = camera::Camera {
            eye: vec3(50., 20., 50.),
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
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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

        // MAP LOAD

        let map = crate::map::new_map();

        let mut instances = vec![];

        const SPACE_BETWEEN: f32 = 2.0;
        for (coords, chunk) in &map.chunks {
            let _3diter = itertools::iproduct!(0..CHUNK_SIZE.0, 0..CHUNK_SIZE.1, 0..CHUNK_SIZE.2);

            let mut i = _3diter
                .filter_map(|(x, y, z)| {
                    if let Block::Air = sl3get(&chunk.blocks, x, y, z) {
                        return None;
                    }

                    let chunk_offset = coords.as_vec2() * (SPACE_BETWEEN * CHUNK_SIZE.0 as f32);

                    let mapping = |n| SPACE_BETWEEN * (n as f32 - CHUNK_SIZE.0 as f32 / 2.0);
                    let position = vec3(
                        mapping(x) + chunk_offset.x,
                        -mapping(y),
                        mapping(z) + chunk_offset.y,
                    );

                    // this is needed so an object at (0, 0, 0) won't get scaled to zero
                    // as Quaternions can affect scale if they're not created correctly
                    // let rotation = match position.try_normalize() {
                    //     Some(position) => Quat::from_axis_angle(position, 45.0),
                    //     _ => Quat::from_axis_angle(Vec3::Z, 0.0),
                    // };
                    let rotation = Quat::from_axis_angle(Vec3::Y, 0.0);

                    Some(Instance { position, rotation })
                })
                .collect::<Vec<_>>();

            instances.append(&mut i);
        }

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let obj_model = resources::load_model(
            "blender_default_cube.obj",
            &device,
            &queue,
            &texture_bind_group_layout,
        )
        .await
        .unwrap();

        let light_uniform = LightUniform::new(Vec3::splat(90.0).with_y(40.0), vec3(1.0, 1.0, 0.5));

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light VB"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: None,
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        // New stuff goes above here!
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                wgpu::PolygonMode::Fill,
            )
        };

        //let fill_pipeline = render_pipeline(wgpu::PolygonMode::Fill);
        let wireframe_render_pipeline = if (device.features() & wgpu::Features::POLYGON_MODE_LINE)
            == wgpu::Features::POLYGON_MODE_LINE
        {
            Some({
                let shader = wgpu::ShaderModuleDescriptor {
                    label: Some("Normal Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
                };
                create_render_pipeline(
                    &device,
                    &render_pipeline_layout,
                    surface_config.format,
                    Some(texture::Texture::DEPTH_FORMAT),
                    &[model::ModelVertex::desc(), InstanceRaw::desc()],
                    shader,
                    wgpu::PolygonMode::Line,
                )
            })
        } else {
            None
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
                wgpu::PolygonMode::Fill,
            )
        };

        Self {
            surface,
            device,
            queue,
            surface_config,
            depth_texture,
            map,
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
            render_pipelines: RenderPipelines {
                camera: render_pipeline,
                camera_wireframe: wireframe_render_pipeline,
                light: light_render_pipeline,
            },
            object: ObjectState {
                model: obj_model,
                instances,
                instance_buffer,
                remake: false,
            },
            light: LightState {
                uniform: light_uniform,
                buffer: light_buffer,
                bind_group: light_bind_group,
            },
        }
    }

    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.camera.positioning.aspect =
            self.surface_config.width as f32 / self.surface_config.height as f32;
        self.surface.configure(&self.device, &self.surface_config);

        self.depth_texture = texture::Texture::create_depth_texture(
            &self.device,
            &self.surface_config,
            "depth_texture",
        );
    }

    pub fn update_instance_buf(&mut self) {
        let mut instances = vec![];

        const SPACE_BETWEEN: f32 = 2.0;
        for (coords, chunk) in &self.map.chunks {
            let _3diter = itertools::iproduct!(0..CHUNK_SIZE.0, 0..CHUNK_SIZE.1, 0..CHUNK_SIZE.2);

            let mut i = _3diter
                .filter_map(|(x, y, z)| {
                    if let Block::Air = sl3get(&chunk.blocks, x, y, z) {
                        return None;
                    }

                    let chunk_offset = coords.as_vec2() * (SPACE_BETWEEN * CHUNK_SIZE.0 as f32);

                    let mapping = |n| SPACE_BETWEEN * (n as f32 - CHUNK_SIZE.0 as f32 / 2.0);
                    let position = vec3(
                        mapping(x) + chunk_offset.x,
                        -mapping(y),
                        mapping(z) + chunk_offset.y,
                    );

                    let rotation = Quat::from_axis_angle(Vec3::Y, 0.0);

                    Some(Instance { position, rotation })
                })
                .collect::<Vec<_>>();

            instances.append(&mut i);
        }

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            });
        self.object.instances = instances;
        self.object.instance_buffer = instance_buffer;
        self.object.remake = false;
    }

    pub(crate) fn render(
        &mut self,
        egui: &mut Option<EguiRenderer>,
        window: Arc<Window>,
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

        use crate::gfx::model::DrawLight;
        render_pass.set_pipeline(&self.render_pipelines.light);
        render_pass.draw_light_model(
            &self.object.model,
            &self.camera.bind_group,
            &self.light.bind_group,
        );

        use crate::gfx::model::DrawModel;
        render_pass.set_pipeline(
            self.interact
                .wireframe
                .then_some(self.render_pipelines.camera_wireframe.as_ref())
                .flatten()
                .unwrap_or(&self.render_pipelines.camera),
        );

        render_pass.draw_model_instanced(
            &self.object.model,
            0..self.object.instances.len() as u32,
            &self.camera.bind_group,
            &self.light.bind_group,
        );

        // drop render pass before we submit to drop the mut borrow on encoder
        drop(render_pass);

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

            egui.update(self);

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

    pub fn update(&mut self) {
        self.camera
            .controller
            .update_camera(&mut self.camera.positioning);
        self.camera
            .uniform
            .update_view_proj(&self.camera.positioning);
        self.queue.write_buffer(
            &self.camera.buffer,
            0,
            bytemuck::cast_slice(&[self.camera.uniform]),
        );

        let old_position: Vec3 = self.light.uniform.position;
        self.light.uniform.position =
            Quat::from_axis_angle(vec3(0.0, 1.0, 0.0), 0.01) * old_position;
        self.queue.write_buffer(
            &self.light.buffer,
            0,
            bytemuck::cast_slice(&[self.light.uniform]),
        );
        if self.object.remake {
            self.update_instance_buf();
        }
    }

    pub fn input(&mut self, event: &WindowEvent, window_size: PhysicalSize<u32>) -> bool {
        self.camera.controller.process_events(event);

        // Deprecated! Replaced with EGUI debug ui.
        // match event {
        //     WindowEvent::CursorMoved {
        //         device_id: _,
        //         position,
        //     } => {}
        //     WindowEvent::KeyboardInput {
        //         event:
        //             KeyEvent {
        //                 state,
        //                 physical_key: PhysicalKey::Code(keycode),
        //                 ..
        //             },
        //         ..
        //     } => {
        //         let is_pressed = *state == ElementState::Pressed;
        //         if *keycode == KeyCode::KeyL && is_pressed {
        //             self.interact.wireframe = !self.interact.wireframe;
        //             return true;
        //         }
        //     }
        //     _ => {}
        // }

        false
    }
}
