#![allow(rust_analyzer::inactive_code)]

mod app;
mod concurrency;
mod gfx;
mod gui;
mod world;

use glam::{Mat3, Quat, Vec3};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wasm_bindgen::UnwrapThrowExt;
use winit::event_loop::EventLoop;
use world::chunk::preload_chunk_cache;

struct Instance {
    position: Vec3,
    rotation: Quat,
}

#[cfg(not(target_arch = "wasm32"))]
type ConnectionOnlyOnNative = rusqlite::Connection;

#[cfg(target_arch = "wasm32")]
type ConnectionOnlyOnNative = ();

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (glam::Mat4::from_translation(self.position)
                * glam::Mat4::from_quat(self.rotation))
            .to_cols_array_2d(),
            normal: Mat3::from_quat(self.rotation).to_cols_array_2d(),
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

fn init_logger() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Debug).expect("Couldn't initialize logger");
        } else {
            //env_logger::init();
            env_logger::builder().filter_level(log::LevelFilter::Warn).init();
        }
    }
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    init_logger();

    log::info!("Hello world!");
    // preload_chunk_cache();

    let conn = rusqlite::Connection::open("./save.sqlite").unwrap();
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS chunks (
            x INTEGER,
            y INTEGER,
            z INTEGER,
            data BLOB,
            PRIMARY KEY (x,y,z)
        )
    "#,
        (),
    )
    .unwrap();

    let event_loop = EventLoop::with_user_event().build().unwrap_throw();

    let mut app = app::Application::new(&event_loop, "BL0CK", conn);
    event_loop.run_app(&mut app).unwrap();
}
