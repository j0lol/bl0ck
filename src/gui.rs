use egui::{FontId, RichText};
use egui_winit::EventResponse;
use glam::ivec3;
use winit::window::Window;

use crate::{
    gfx::Gfx,
    world::{
        chunk::{Chunk, ChunkScramble},
        World,
    },
};

const FPS_AVG_WINDOW: usize = 120;
pub struct EguiRenderer {
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    frame_started: bool,
    pub scale_factor: f32,
    pub chunk_influence: (i32, i32, i32),
    pub frame_count: u64,
    pub fps_average: [f64; FPS_AVG_WINDOW],
}

impl EguiRenderer {
    // Just a helper
    pub fn ctx(&self) -> &egui::Context {
        self.state.egui_ctx()
    }

    pub fn new(
        device: &wgpu::Device,
        output_color_format: wgpu::TextureFormat,
        output_depth_format: Option<wgpu::TextureFormat>,
        msaa_samples: u32,
        window: &Window,
    ) -> EguiRenderer {
        let context = egui::Context::default();
        let state = egui_winit::State::new(
            context,
            egui::viewport::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(2 * 1024), // "max texture side"  of 2048
        );

        let renderer = egui_wgpu::Renderer::new(
            device,
            output_color_format,
            output_depth_format,
            msaa_samples,
            true,
        );

        #[cfg(target_arch = "wasm32")]
        {
            state.egui_ctx().set_pixels_per_point(1.0);
            state.egui_ctx().set_zoom_factor(1.0);
        }

        EguiRenderer {
            state,
            renderer,
            frame_started: false,
            scale_factor: 1.0,
            chunk_influence: (0, 0, 0),
            frame_count: 0,
            fps_average: [0.; FPS_AVG_WINDOW],
        }
    }

    pub fn handle_input(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        let EventResponse { consumed, .. } = self.state.on_window_event(window, event);
        consumed
    }

    pub(crate) fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.state.take_egui_input(window);
        self.ctx().begin_pass(raw_input);
        self.frame_started = true;
    }

    pub(crate) fn end_frame_and_draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        window: &Window,
        surface_view: &wgpu::TextureView,
        screen_descriptor: egui_wgpu::ScreenDescriptor,
    ) {
        if !self.frame_started {
            panic!("begin_frame must be called before end_frame_and_draw can be called!");
        }

        #[cfg(not(target_arch = "wasm32"))]
        self.ctx()
            .set_pixels_per_point(screen_descriptor.pixels_per_point);

        let full_output = self.ctx().end_pass();

        self.state
            .handle_platform_output(window, full_output.platform_output);

        #[cfg(not(target_arch = "wasm32"))]
        let tris = self
            .ctx()
            .tessellate(full_output.shapes, self.ctx().pixels_per_point());

        #[cfg(target_arch = "wasm32")]
        let tris = self.ctx().tessellate(full_output.shapes, 1.0);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }

        self.renderer
            .update_buffers(device, queue, encoder, &tris, &screen_descriptor);

        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("EGUI Main Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        self.renderer.render(
            &mut render_pass.forget_lifetime(),
            &tris,
            &screen_descriptor,
        );
        for x in &full_output.textures_delta.free {
            self.renderer.free_texture(x)
        }

        self.frame_started = false;
    }

    pub fn update(&mut self, gfx: &mut Gfx, world: &mut World, dt: instant::Duration) {
        let mut scale_factor = self.scale_factor;
        let (mut chunk_x, mut chunk_y, mut chunk_z) = self.chunk_influence;
        let (mut grid_x, mut grid_y, mut grid_z) = world.map.chunks.offset();
        let mut camera_load = gfx.camera.controller.load_chunks;

        let dt = dt.as_secs_f32();
        self.frame_count += 1;
        self.fps_average[(self.frame_count % FPS_AVG_WINDOW as u64) as usize] = 1.0_f64 / dt as f64;
        let mean = self.fps_average.iter().sum::<f64>() / FPS_AVG_WINDOW as f64;

        let ctx = self.ctx();
        egui::Window::new("Debug Menu")
            .resizable(true)
            .vscroll(true)
            .default_open(false)
            .show(ctx, |ui| {
                ui.heading("Performance debugging stats...");
                ui.label(format!(
                    "FPS: {:.1} (smoothed over an interval of {})",
                    mean, FPS_AVG_WINDOW
                ));
                ui.label(format!("FPS: {:.1} (jittery)", 1.0 / dt));
                ui.label(format!("Instances: {:?}", gfx.object.instances.len()));
                // ui.label(format!(
                //     "Vertices (guess): {:?}",
                //     gfx.object.instances.len() as u32
                //         * gfx
                //             .object
                //             .model
                //             .meshes
                //             .iter()
                //             .map(|x| x.num_elements)
                //             .sum::<u32>()
                // ));

                ui.separator();

                ui.heading("Debugging toys...");
                ui.horizontal(|ui| {
                    ui.label("Draw Color");

                    // Absolutely disgusting code!
                    // I would need a func to convert between wgpu::Color (f64 rgb)
                    // to f32 rgb and back. idk
                    //
                    // Better yet, a generic color type that can convert to wgpu
                    // and others
                    let c = gfx.interact.clear_color;
                    let mut color = [c.r as _, c.g as _, c.b as _];

                    ui.color_edit_button_rgb(&mut color);
                    gfx.interact.clear_color.r = color[0] as _;
                    gfx.interact.clear_color.g = color[1] as _;
                    gfx.interact.clear_color.b = color[2] as _;
                });

                ui.checkbox(&mut gfx.interact.wireframe, "Wireframe (PC Only)");

                ui.separator();

                ui.label(format!("Cam pos: {:?}", gfx.camera.object.position));
                ui.label(format!("Cam yaw: {:?}", gfx.camera.object.yaw));
                ui.label(format!("Cam pitch: {:?}", gfx.camera.object.pitch));

                ui.add(
                    egui::Slider::new(&mut gfx.camera.controller.speed, 0.1..=1000.0)
                        .text("Cam Speed")
                        .logarithmic(true),
                );

                ui.separator();

                ui.label("Camera input \"bitfield\":");
                ui.label(
                    RichText::new(format!("{:?}", gfx.camera.controller.movement))
                        .font(FontId::monospace(11.0)),
                );
                ui.label(format!(
                    "... which is a movement vector of: {:?}",
                    gfx.camera.controller.movement.vec3()
                ));

                ui.separator();

                ui.add(
                    egui::Slider::new(&mut gfx.interact.sun_speed, 0.1..=100.0)
                        .text("Sun rotational speed (radians per second)")
                        .logarithmic(true),
                );

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(format!("Pixels per point: {}", ctx.pixels_per_point()));
                    if ui.button("-").clicked() {
                        scale_factor = (scale_factor - 0.1).max(0.3);
                    }
                    if ui.button("+").clicked() {
                        scale_factor = (scale_factor + 0.1).min(3.0);
                    }
                });

                ui.separator();

                ui.heading("World toys...");

                ui.checkbox(&mut camera_load, "Camera position loads chunks");
                ui.label("Move chunk window... ");
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        !camera_load,
                        egui::DragValue::new(&mut grid_x)
                            .speed(0.1)
                            .update_while_editing(false),
                    );
                    ui.label("x ");

                    ui.add_enabled(
                        !camera_load,
                        egui::DragValue::new(&mut grid_y)
                            .speed(0.1)
                            .update_while_editing(false),
                    );
                    ui.label("y ");

                    ui.add_enabled(
                        !camera_load,
                        egui::DragValue::new(&mut grid_z)
                            .speed(0.1)
                            .update_while_editing(false),
                    );
                    ui.label("z.");
                });

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Scramble chunk at...");
                    ui.add(egui::DragValue::new(&mut chunk_x).speed(0.1));
                    ui.label("x ");

                    ui.add(egui::DragValue::new(&mut chunk_y).speed(0.1));
                    ui.label("y ");

                    ui.add(egui::DragValue::new(&mut chunk_z).speed(0.1));
                    ui.label("z.");
                });

                ui.horizontal(|ui| {
                    let pos = ivec3(chunk_x, chunk_y, chunk_z);

                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("Random").clicked() {
                        let c = Chunk::generate(pos, ChunkScramble::Random);
                        world.map.chunks.set(pos.into(), c);
                        gfx.object.remake = true;
                    }
                    if ui.button("Normal").clicked() {
                        let c = Chunk::generate(pos, ChunkScramble::Normal);
                        world.map.chunks.set(pos.into(), c);
                        gfx.object.remake = true;
                    }
                    if ui.button("Inverse").clicked() {
                        let c = Chunk::generate(pos, ChunkScramble::Inverse);
                        world.map.chunks.set(pos.into(), c);
                        gfx.object.remake = true;
                    }
                });

                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        world.save().unwrap();
                    }
                });
            });

        self.scale_factor = scale_factor;
        self.chunk_influence = (chunk_x, chunk_y, chunk_z);

        gfx.camera.controller.load_chunks = camera_load;

        if !camera_load {
            if (grid_x, grid_y, grid_z) != world.map.chunks.offset() {
                world
                    .map
                    .chunks
                    .reposition((grid_x, grid_y, grid_z), |_old, new, chunk| {
                        *chunk = Chunk::load(ivec3(new.0, new.1, new.2)).unwrap();
                    });
                gfx.object.remake = true;
            }
        }
    }
}
