use glam::{Mat4, Vec3, Vec4};
use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    keyboard::KeyCode,
    keyboard::PhysicalKey,
};

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);

        proj * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_position: Vec4,
    view_proj: Mat4,
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: Vec4::splat(0.0),
            view_proj: Mat4::IDENTITY,
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_position = camera.eye.extend(1.0);
        self.view_proj = camera.build_view_projection_matrix();
    }
}

pub struct CameraController {
    pub speed: f32,
    buttons: glam::BVec4, // fwd, lft, bwd, rht
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            buttons: glam::BVec4::FALSE,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
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
                match keycode {
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        self.buttons.x = is_pressed;
                        true
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.buttons.y = is_pressed;
                        true
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.buttons.z = is_pressed;
                        true
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.buttons.w = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.length();

        // Prevents glitching when the camera gets too close to the
        // center of the scene.
        if self.buttons.x && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.buttons.z {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.length();

        if self.buttons.w {
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.buttons.y {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}
