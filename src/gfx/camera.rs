use glam::{ivec3, vec3, vec4, IVec3, Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use instant::Duration;
use itertools::Itertools;
use rollgrid::math::Convert;
use std::f32::consts::FRAC_2_PI;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent},
    keyboard::KeyCode,
    keyboard::PhysicalKey,
};

use crate::world::{chunk::Chunk, World};

use super::Gfx;

const MAX_CAMERA_PITCH: f32 = (3.0 / std::f32::consts::PI) - 0.0001;

type Rad = f32;
type Distance = f32;

#[derive(Default, Debug)]
pub struct AxisInput {
    x: bool,
    x_neg: bool,
    y: bool,
    y_neg: bool,
    z: bool,
    z_neg: bool,
}

impl AxisInput {
    pub fn vec3(&self) -> Vec3 {
        let x = match (self.x, self.x_neg) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        };

        let y = match (self.y, self.y_neg) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        };

        let z = match (self.z, self.z_neg) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        };
        vec3(x, y, z)
    }
}

pub struct Camera {
    pub position: Vec3,
    yaw: Rad,
    pitch: Rad,
}

impl Camera {
    pub fn new(position: Vec3, yaw: Rad, pitch: Rad) -> Self {
        Self {
            position,
            yaw,
            pitch,
        }
    }

    pub fn mat4(&self) -> Mat4 {
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();

        Mat4::look_to_rh(
            self.position,
            vec3(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize(),
            Vec3::Y,
        )
    }
}

pub struct Projection {
    aspect: f32,
    fovy: Rad,
    znear: Distance,
    zfar: Distance,
}

impl Projection {
    pub fn new(resolution: Vec2, fovy: Rad, znear: Distance, zfar: Distance) -> Self {
        Self {
            aspect: resolution.x / resolution.y,
            fovy,
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, resolution: Vec2) {
        self.aspect = resolution.x / resolution.y;
    }

    pub fn mat4(&self) -> Mat4 {
        Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

pub struct CameraController {
    pub(crate) movement: AxisInput,
    pub(crate) rotation: Vec2,
    scroll: f32,
    pub(crate) speed: f32,
    sensitivity: f32,
    pub(crate) load_chunks: bool,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            movement: AxisInput::default(),
            rotation: Vec2::ZERO,
            scroll: 0.,
            speed,
            sensitivity,
            load_chunks: true,
        }
    }

    pub fn process_keyboard(
        &mut self,
        key: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
    ) -> bool {
        let pressed = state == ElementState::Pressed;

        let field = match key {
            KeyCode::KeyW | KeyCode::ArrowUp => &mut self.movement.x,
            KeyCode::KeyS | KeyCode::ArrowDown => &mut self.movement.x_neg,
            KeyCode::KeyD | KeyCode::ArrowRight => &mut self.movement.z,
            KeyCode::KeyA | KeyCode::ArrowLeft => &mut self.movement.z_neg,
            KeyCode::Space => &mut self.movement.y,
            KeyCode::ShiftLeft => &mut self.movement.y_neg,
            _ => {
                return false;
            }
        };
        *field = pressed;

        true
    }

    pub fn process_mouse(&mut self, mouse: Vec2) {
        self.rotation += mouse;
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        let line_px = 100.0;

        self.scroll = -match delta {
            MouseScrollDelta::LineDelta(_, scroll) => scroll * line_px,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => *scroll as f32,
        };
    }

    pub fn update_camera(
        &mut self,
        camera: &mut Camera,
        duration: Duration,
        world: &mut World,
        remake: &mut bool,
    ) {
        let dt = duration.as_secs_f32();
        let movement = self.movement.vec3();

        // X/Z movement
        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();
        let forward = vec3(yaw_cos, 0.0, yaw_sin).normalize();
        let right = vec3(-yaw_sin, 0.0, yaw_cos).normalize();
        camera.position += forward * movement.x * self.speed * dt;
        camera.position += right * movement.z * self.speed * dt;

        // Move toward focal point
        let (pitch_sin, pitch_cos) = camera.pitch.sin_cos();
        let toward_focus = vec3(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position += toward_focus * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        camera.position.y += movement.y * self.speed * dt;

        camera.yaw += self.rotation.x * self.sensitivity * dt;
        camera.pitch += -self.rotation.y * self.sensitivity * dt;

        if self.load_chunks {
            const BLOCK_UNIT_SIZE: i32 = 32;
            let chunk_relative = IVec3::from(
                (camera.position.x as i32 / BLOCK_UNIT_SIZE,
                -(camera.position.y as i32 / BLOCK_UNIT_SIZE),
                camera.position.z as i32 / BLOCK_UNIT_SIZE,
            )) + IVec3::splat(-2);
            if chunk_relative != world.map.chunks.offset().into() {
                world
                    .map
                    .chunks
                    .reposition((IVec3::from(chunk_relative)).into(), |_old, new, chunk| {
                        *chunk = Chunk::load(ivec3(new.0, new.1, new.2)).unwrap();
                    });
                *remake = true;
            }
        }

        self.rotation = Vec2::ZERO;

        // Keep the camera's angle from going too high/low.
        if camera.pitch < -MAX_CAMERA_PITCH {
            camera.pitch = -MAX_CAMERA_PITCH;
        } else if camera.pitch > MAX_CAMERA_PITCH {
            camera.pitch = MAX_CAMERA_PITCH;
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_position: Vec4,
    pub view_proj: Mat4,
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: Vec4::splat(0.0),
            view_proj: Mat4::IDENTITY,
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position = camera.position.extend(1.0);
        self.view_proj = projection.mat4() * camera.mat4();
    }
}

