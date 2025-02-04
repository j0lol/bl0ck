use egui_winit::EventResponse;
use winit::window::Window;

pub struct EguiRenderer {
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    frame_started: bool,
    pub scale_factor: f32,
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

        EguiRenderer {
            state,
            renderer,
            frame_started: false,
            scale_factor: 1.0
        }
    }

    pub fn handle_input(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        let EventResponse {consumed, ..} = self.state.on_window_event(window, event);
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

        self.ctx().set_pixels_per_point(screen_descriptor.pixels_per_point);

        let full_output = self.ctx().end_pass();

        self.state
            .handle_platform_output(window, full_output.platform_output);

        let tris = self
            .ctx()
            .tessellate(full_output.shapes, self.ctx().pixels_per_point());

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

    pub fn update(&mut self) {
        let ctx = self.ctx();

        let mut scale_factor = self.scale_factor;

        egui::Window::new("winit + egui + wgpu says hello!")
            .resizable(true)
            .vscroll(true)
            .default_open(false)
            .show(ctx, |ui| {
                ui.label("Label!");

                if ui.button("Button!").clicked() {
                    println!("boom!")
                }

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
            });

        self.scale_factor = scale_factor;
    }
}