use crate::gfx::{Gfx, GfxBuilder, MaybeGfx};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

pub(crate) const WASM_WIN_SIZE: (u32, u32) = (640, 480);

// TODO citation:
// https://github.com/Bentebent/rita/ for winit 29.0->30.0 migration
// https://github.com/erer1243/wgpu-0.20-winit-0.30-web-example/blob/master/src/lib.rs For better winit 30.0 impl
// thanks everyone. the migration is really counter-intuitive

pub struct Application {
    window_attributes: WindowAttributes,
    gfx_state: MaybeGfx,
    window: Option<Arc<Window>>,
}

impl Application {
    pub fn new(event_loop: &EventLoop<Gfx>, title: &str) -> Self {
        Self {
            window_attributes: Window::default_attributes().with_title(title),
            gfx_state: MaybeGfx::Builder(GfxBuilder::new(event_loop.create_proxy())),
            window: None,
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

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, gfx: Gfx) {
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
            gfx.input(&event, window.inner_size());
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                if matches!(
                    event,
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    }
                ) {
                    event_loop.exit()
                }
            }
            WindowEvent::Resized(physical_size) => {
                gfx.resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
                // Some horrible nesting here! Don't tell Linus...
                if let Some(ref window) = &self.window {
                    window.request_redraw();
                    match gfx.render() {
                        Ok(_) => {
                            gfx.update();
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
