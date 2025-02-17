use std::f32::consts::PI;
use glam::{vec3, Mat4, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightUniform {
    pub view_pos: Vec3,
    _pad: u32,
    pub view_proj: Mat4,
    pub color: Vec3,
    _pad2: u32,
}
impl LightUniform {
    pub fn new(position: Vec3, color: Vec3) -> LightUniform {
        LightUniform {
            view_pos: position,
            view_proj: {
                let view = Mat4::look_at_rh(position, Vec3::ZERO, Vec3::Y);
                let proj = Mat4::perspective_rh(PI / 2., 1.0, 0.1, 1000.0);

                view * proj
            },
            color,
            _pad: 0,
            _pad2: 0,
        }
    }

    pub fn update_view_proj(&mut self, aspect_ratio: f32) {
        self.view_proj = {
            let view = Mat4::look_at_rh(vec3(50., 20., 50.), Vec3::ZERO, Vec3::Y);
            let proj = Mat4::perspective_rh(PI / 2., aspect_ratio, 0.1, 1000.0);

            view * proj
        };
    }

    // pub fn build_view_projection_matrix(&self) -> Mat4 {
    // let view = Mat4::look_at_rh(self.eye, self.target, self.up);
    // let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);

    //     proj * view
    // }
}
