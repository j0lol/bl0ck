use egui_winit::EventResponse;
use glam::{ivec2, ivec3, IVec2};
use winit::window::Window;

use crate::{
    gfx::Gfx,
    world::map::{chunk_scramble_dispatch, ChunkScramble},
    world::World,
};

pub struct EguiRenderer {
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    frame_started: bool,
    pub scale_factor: f32,
    pub chunk_influence: (i32, i32),
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
            chunk_influence: (0, 0),
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

    pub fn update(&mut self, gfx: &mut Gfx, world: &mut World) {
        let ctx = self.ctx();

        let mut scale_factor = self.scale_factor;
        let (mut chunk_x, mut chunk_z) = self.chunk_influence;

        egui::Window::new("Debug Menu")
            .resizable(true)
            .vscroll(true)
            .default_open(false)
            .show(ctx, |ui| {
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

                ui.add(
                    egui::Slider::new(&mut gfx.camera.controller.speed, 0.1..=10.0)
                        .text("Cam Speed"),
                );

                ui.separator();

                ui.label(format!(
                    "The camera is pointing at {:?}",
                    gfx.camera.object.target
                ));
                ui.add(
                    egui::Slider::new(&mut gfx.camera.object.eye.x, -500.0..=500.0).text("Cam X"),
                );
                ui.add(
                    egui::Slider::new(&mut gfx.camera.object.eye.y, -500.0..=500.0).text("Cam Y"),
                );
                ui.add(
                    egui::Slider::new(&mut gfx.camera.object.eye.z, -500.0..=500.0).text("Cam Z"),
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

                ui.horizontal(|ui| {
                    ui.label("Scramble chunk at...");
                    ui.add(egui::DragValue::new(&mut chunk_x).speed(0.1));
                    ui.label("x, ");

                    ui.add(egui::DragValue::new(&mut chunk_z).speed(0.1));
                    ui.label("z. ");
                });

                ui.horizontal(|ui| {
                    let pos = ivec3(chunk_x, 0, chunk_z);

                    if ui.button("Random").clicked() {
                        let c = chunk_scramble_dispatch(ChunkScramble::Random)(pos);
                        world.map.chunks.insert(pos.into(), c);
                        gfx.object.remake = true;
                    }
                    if ui.button("Normal").clicked() {
                        let c = chunk_scramble_dispatch(ChunkScramble::Normal)(pos);
                        world.map.chunks.insert(pos.into(), c);
                        gfx.object.remake = true;
                    }
                    if ui.button("Inverse").clicked() {
                        let c = chunk_scramble_dispatch(ChunkScramble::Inverse)(pos);
                        world.map.chunks.insert(pos.into(), c);
                        gfx.object.remake = true;
                    }
                });

                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        world.save().unwrap();
                    }
                    if ui.button("Load").clicked() {
                        *world = World::load().unwrap();
                        gfx.object.remake = true;
                    }
                });
            });

        self.scale_factor = scale_factor;
        self.chunk_influence = (chunk_x, chunk_z);
    }
}
