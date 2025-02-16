use crate::{
    gfx::{Gfx, GfxBuilder, MaybeGfx},
    gui::EguiRenderer,
    world::map::new,
    world::World,
};
use glam::{dvec2, vec2};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

pub(crate) const WASM_WIN_SIZE: (u32, u32) = (640 * 2, 480 * 2);

// TODO citation:
// https://github.com/Bentebent/rita/ for winit 29.0->30.0 migration
// https://github.com/erer1243/wgpu-0.20-winit-0.30-web-example/blob/master/src/lib.rs For better winit 30.0 impl
// thanks everyone. the migration is really counter-intuitive

pub struct Application {
    window_attributes: WindowAttributes,
    gfx_state: MaybeGfx,
    window: Option<Arc<Window>>,
    egui: Option<EguiRenderer>,
    world: World,
    last_render_time: instant::Instant,
}

impl Application {
    pub fn new(event_loop: &EventLoop<Gfx>, title: &str) -> Self {
        Self {
            window_attributes: Window::default_attributes().with_title(title),
            gfx_state: MaybeGfx::Builder(GfxBuilder::new(event_loop.create_proxy())),
            window: None,
            egui: None,
            world: World { map: new() },
            last_render_time: instant::Instant::now(),
        }
    }
}

impl ApplicationHandler<Gfx> for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        let window = Arc::new(
            event_loop
                .create_window(self.window_attributes.clone())
                .unwrap(),
        );

        if let MaybeGfx::Builder(builder) = &mut self.gfx_state {
            builder.build_and_send(window.clone());
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Winit prevents sizing with CSS, so we have to set
            // the size manually when on web.
            use winit::dpi::PhysicalSize;
            let _ = window.request_inner_size(PhysicalSize::new(WASM_WIN_SIZE.0, WASM_WIN_SIZE.1));

            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| {
                    let dst = doc.get_element_by_id("wasm-example")?;
                    let canvas = web_sys::Element::from(window.canvas()?);
                    dst.append_child(&canvas).ok()?;
                    Some(())
                })
                .expect("Couldn't append canvas to document body.");
        }

        window.request_redraw();

        self.window = Some(window);
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let (MaybeGfx::Graphics(gfx), DeviceEvent::MouseMotion { delta }) =
            (&mut self.gfx_state, event)
        {
            if gfx.camera.mouse_focused {
            gfx.camera
                .controller
                .process_mouse(dvec2(delta.0, delta.1).as_vec2())
            }
        }
    }
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, gfx: Gfx) {
        if let Some(window) = &self.window {
            let egui = EguiRenderer::new(&gfx.device, gfx.surface_config.format, None, 1, window);
            self.egui = Some(egui);
        }
        self.gfx_state = MaybeGfx::Graphics(gfx);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let MaybeGfx::Graphics(gfx) = &mut self.gfx_state else {
            if let (WindowEvent::RedrawRequested, Some(ref window)) = (event, &self.window) {
                window.request_redraw();
            }
            return;
        };

        if let Some(ref window) = &self.window {
            // Returns true if EGUI consumes the input.
            if self
                .egui
                .as_mut()
                .map(|egui| egui.handle_input(window, &event))
                .is_some_and(std::convert::identity)
            {
                return;
            }

            gfx.input(&event, window.inner_size());
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::MouseInput {
                button: winit::event::MouseButton::Left,
                state,
                ..
            } => {
                if !gfx.camera.mouse_focused {
                    gfx.camera.mouse_focused = true;
                    if let Some(ref window) = &self.window {
                        match window.set_cursor_grab(winit::window::CursorGrabMode::Locked) {
                            Ok(()) => {
                                window.set_cursor_visible(false);
                            }
                            Err(e) => {
                                log::error!("{e}");
                            }
                        }
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => match event {
                KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::KeyX),
                    ..
                } => event_loop.exit(),

                KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                    ..
                } => {
                    if let Some(ref window) = &self.window {
                        if gfx.camera.mouse_focused {
                            gfx.camera.mouse_focused = false;

                            match window.set_cursor_grab(winit::window::CursorGrabMode::None) {
                                Ok(()) => {
                                    window.set_cursor_visible(true);
                                }
                                Err(e) => {
                                    log::error!("{e}");
                                }
                            }
                        }
                    }
                }

                _ => {}
            },
            WindowEvent::Resized(physical_size) => {
                gfx.resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
                // Some horrible nesting here! Don't tell Linus...
                if let Some(ref window) = &self.window {
                    let now = instant::Instant::now();
                    let dt = now - self.last_render_time;
                    self.last_render_time = now;

                    window.request_redraw();
                    match gfx.render(&mut self.egui, window.clone(), &mut self.world, dt) {
                        Ok(_) => {
                            // TODO CITE https://github.com/kaphula/winit-egui-wgpu-template/blob/master/src/app.rs#L3
                            gfx.update(&mut self.world, dt);
                        }
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            gfx.resize(window.inner_size());
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of memory!");
                            event_loop.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            log::warn!("Surface timeout!");
                        }
                        Err(wgpu::SurfaceError::Other) => {
                            log::error!("Other surface-error!");
                            event_loop.exit();
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
