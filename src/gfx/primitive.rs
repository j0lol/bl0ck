// TODO CITE https://github.com/whoisryosuke/wgpu-hello-world/blob/play/primitives-model-test/src/primitives/mod.rs

use crate::gfx::model::ModelVertex;
use crate::gfx::primitive::cube::{cube_indices, cube_vertices, Faces};
use crate::gfx::resources::load_texture;
use crate::gfx::{model, primitive};
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue};

pub(crate) mod cube {
    use crate::gfx::model::ModelVertex;
    use itertools::Itertools;

    #[derive(Copy, Clone, Default)]
    pub struct Faces {
        front: bool,
        back: bool,
        top: bool,
        bottom: bool,
        right: bool,
        left: bool,
    }
    impl Faces {
        pub(crate) const ALL: Faces = Faces {
            front: true,
            back: true,
            top: true,
            bottom: true,
            right: true,
            left: true,
        };
        const NONE: Faces = Faces {
            front: false,
            back: false,
            top: false,
            bottom: false,
            right: false,
            left: false,
        };

        fn arr(
            Faces {
                front,
                back,
                top,
                bottom,
                right,
                left,
            }: Self,
        ) -> [bool; 6] {
            [front, back, top, bottom, left, right]
        }
    }

    pub fn cube_vertices(faces: Faces, scale: f32, (x, y, z): (f32, f32, f32)) -> Vec<ModelVertex> {
        const P: f32 = 1.0;
        const N: f32 = -1.0;
        let front = vec![
            ModelVertex {
                position: [N, N, P],
                normal: [0.0, 0.0, P],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, N, P],
                normal: [0.0, 0.0, N],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, P, P],
                normal: [P, 0.0, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [N, P, P],
                normal: [N, 0.0, 0.0],
                tex_coords: [0.0, 0.5],
            },
        ];
        let back = vec![
            ModelVertex {
                position: [N, N, N],
                normal: [0.0, P, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [N, P, N],
                normal: [0.0, N, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, P, N],
                normal: [0.0, 0.0, P],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, N, N],
                normal: [0.0, 0.0, N],
                tex_coords: [0.0, 0.5],
            },
        ];
        let top = vec![
            ModelVertex {
                position: [N, P, N],
                normal: [P, 0.0, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [N, P, P],
                normal: [N, 0.0, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, P, P],
                normal: [0.0, P, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, P, N],
                normal: [0.0, N, 0.0],
                tex_coords: [0.0, 0.5],
            },
        ];
        let bottom = vec![
            ModelVertex {
                position: [N, N, N],
                normal: [0.0, 0.0, P],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, N, N],
                normal: [0.0, 0.0, N],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, N, P],
                normal: [P, 0.0, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [N, N, P],
                normal: [N, 0.0, 0.0],
                tex_coords: [0.0, 0.5],
            },
        ];
        let right = vec![
            ModelVertex {
                position: [P, N, N],
                normal: [0.0, P, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, P, N],
                normal: [0.0, N, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, P, P],
                normal: [0.0, 0.0, P],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [P, N, P],
                normal: [0.0, 0.0, N],
                tex_coords: [0.0, 0.5],
            },
        ];
        let left = vec![
            ModelVertex {
                position: [N, N, N],
                normal: [P, 0.0, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [N, N, P],
                normal: [N, 0.0, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [N, P, P],
                normal: [0.0, P, 0.0],
                tex_coords: [0.0, 0.5],
            },
            ModelVertex {
                position: [N, P, N],
                normal: [0.0, N, 0.0],
                tex_coords: [0.0, 0.5],
            },
        ];

        Faces::arr(faces)
            .iter()
            .zip([front, back, top, bottom, right, left])
            .filter_map(|(face, x)| face.then_some(x))
            .flatten()
            .map(|mut vertex| {
                vertex.position = [
                    vertex.position[0] + x * scale,
                    vertex.position[1] + y * scale,
                    vertex.position[2] + z * scale,
                ];
                vertex.normal = [
                    vertex.normal[0] * scale,
                    vertex.normal[1] * scale,
                    vertex.normal[2] * scale,
                ];
                vertex
            })
            .collect_vec()
    }

    pub fn cube_indices(n: u32, faces: Faces) -> Vec<u32> {
        let vertices_count = cube_vertices(faces, 0., (0., 0., 0.)).len() as u32;
        let front = vec![0, 1, 2, 0, 2, 3];
        let back = vec![4, 5, 6, 4, 6, 7];
        let top = vec![8, 9, 10, 8, 10, 11];
        let bottom = vec![12, 13, 14, 12, 14, 15];
        let right = vec![16, 17, 18, 16, 18, 19];
        let left = vec![20, 21, 22, 20, 22, 23];

        Faces::arr(faces)
            .iter()
            .zip([front, back, top, bottom, right, left])
            .filter_map(|(face, x)| face.then_some(x))
            .flatten()
            .map(|i| i + (n * vertices_count))
            .collect_vec()
    }
}

pub struct PrimitiveMesh {
    pub model: model::Model,
}

impl PrimitiveMesh {
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        vertices: &[ModelVertex],
        indices: &Vec<u32>,
    ) -> Self {
        let primitive_type = "Cube";

        println!("[PRIMITIVE] Creating cube materials");

        let mut materials = Vec::new();
        let diffuse_texture = load_texture(String::from("cat_face.png"), device, queue)
            .await
            .expect("Couldn't load placeholder texture for primitive");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: None,
        });

        materials.push(model::Material {
            name: primitive_type.to_string(),
            diffuse_texture,
            bind_group,
        });

        println!("[PRIMITIVE] Creating cube mesh buffers");
        let mut meshes = Vec::new();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Vertex Buffer", primitive_type)),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Index Buffer", primitive_type)),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        meshes.push(model::Mesh {
            name: primitive_type.to_string(),
            vertex_buffer,
            index_buffer,
            num_elements: indices.len() as u32,
            material: 0,
        });

        let model = model::Model { meshes, materials };

        Self { model }
    }
}

pub struct PrimitiveMeshBuilder {
    vertices: Vec<ModelVertex>,
    indices: Vec<u32>,
    objects: u32,
}

impl PrimitiveMeshBuilder {
    pub fn new() -> Self {
        PrimitiveMeshBuilder {
            vertices: Default::default(),
            indices: Default::default(),
            objects: 0,
        }
    }
    pub fn cube(mut self, faces: Faces, x: f32, y: f32, z: f32) -> Self {
        self.vertices
            .append(&mut cube_vertices(faces, 1.0, (x, y, z)));
        self.indices.append(&mut cube_indices(self.objects, faces));
        self.objects += 1;
        self
    }

    pub async fn build(
        self,
        device: &Device,
        queue: &Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Option<PrimitiveMesh> {
        if self.objects == 0 {
            return None;
        }

        let mut materials = Vec::new();
        let diffuse_texture = load_texture(String::from("cat_face.png"), device, queue)
            .await
            .expect("Couldn't load placeholder texture for primitive");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: None,
        });

        materials.push(model::Material {
            name: "Primitive".to_string(),
            diffuse_texture,
            bind_group,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&"Primitive Vertex Buffer".to_string()),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&"Primitive Index Buffer".to_string()),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut meshes = Vec::new();
        meshes.push(model::Mesh {
            name: "Primitive".to_string(),
            vertex_buffer,
            index_buffer,
            num_elements: self.indices.len() as u32,
            material: 0,
        });

        let model = model::Model { meshes, materials };

        Some(PrimitiveMesh { model })
    }
}
