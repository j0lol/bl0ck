use super::texture;
use std::ops::Range;

pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

pub struct Material {
    pub name: String, // for debugging!
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

#[derive(Debug)]
pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        bind_groups: &[&'a wgpu::BindGroup],
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        bind_groups: &[&'a wgpu::BindGroup],
    );
    fn draw_model(&mut self, model: &'a Model, bind_groups: &[&'a wgpu::BindGroup]);
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        bind_groups: &[&'a wgpu::BindGroup],
    );
}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        bind_groups: &[&'b wgpu::BindGroup],
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, bind_groups);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'a Material,
        instances: Range<u32>,
        bind_groups: &[&'b wgpu::BindGroup],
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, &material.bind_group, &[]);
        for (i, group) in bind_groups.iter().enumerate() {
            self.set_bind_group((i + 1) as u32, *group, &[]);
        }
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }
    fn draw_model(&mut self, model: &'b Model, bind_groups: &[&'b wgpu::BindGroup]) {
        self.draw_model_instanced(model, 0..1, bind_groups);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        bind_groups: &[&'b wgpu::BindGroup],
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), bind_groups);
        }
    }
}

pub trait DrawLight<'a> {
    fn draw_light_mesh(&mut self, mesh: &'a Mesh, bind_groups: &[&'a wgpu::BindGroup]);
    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
        bind_groups: &[&'a wgpu::BindGroup],
    );

    fn draw_light_model(&mut self, model: &'a Model, bind_groups: &[&'a wgpu::BindGroup]);
    fn draw_light_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        bind_groups: &[&'a wgpu::BindGroup],
    );
}

impl<'a, 'b> DrawLight<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_light_mesh(&mut self, mesh: &'b Mesh, bind_groups: &[&'b wgpu::BindGroup]) {
        self.draw_light_mesh_instanced(mesh, 0..1, bind_groups);
    }

    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        instances: Range<u32>,
        bind_groups: &[&'b wgpu::BindGroup],
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        for (i, group) in bind_groups.iter().enumerate() {
            self.set_bind_group(i as u32, *group, &[]);
        }
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_light_model(&mut self, model: &'b Model, bind_groups: &[&'b wgpu::BindGroup]) {
        self.draw_light_model_instanced(model, 0..1, bind_groups);
    }
    fn draw_light_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        bind_groups: &[&'b wgpu::BindGroup],
    ) {
        for mesh in &model.meshes {
            self.draw_light_mesh_instanced(mesh, instances.clone(), bind_groups);
        }
    }
}
