use super::{model, texture};
use crate::gfx::model::{Material, Mesh};
use crate::gfx::texture::Texture;
use cfg_if::cfg_if;
use std::error::Error;
use std::io::{BufReader, Cursor};
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, Device, Queue};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    // let mut origin = location.pathname().unwrap();
    // if !origin.ends_with("res") {
    //     origin = format!("{}/res", origin);
    // }
    // let base = reqwest::Url::parse(&format!("{}/", origin)).unwrap();
    // base.join(file_name).unwrap()

    reqwest::Url::parse(&format!(
        "{}/../res/{}",
        location.href().unwrap(),
        file_name
    ))
    .unwrap()
}

pub trait ProcessingModel {
    fn positions(&self) -> Vec<f32>;
    fn normals(&self) -> Vec<f32>;
    fn indices(&self) -> Vec<u32>;
    fn tex_coords(&self) -> Vec<f32>;

    fn tri_positions(&self, tri: usize) -> [f32; 3] {
        let positions = self.positions();
        [
            positions[tri * 3],
            positions[tri * 3 + 1],
            positions[tri * 3 + 2],
        ]
    }

    fn tri_normals(&self, tri: usize) -> [f32; 3] {
        let normals = self.normals();
        if normals.is_empty() {
            glam::Vec3::ZERO.into()
        } else {
            [normals[tri * 3], normals[tri * 3 + 1], normals[tri * 3 + 2]]
        }
    }

    fn mesh(&self, device: &wgpu::Device, material_id: usize) -> Mesh {
        let vertices = (0..self.positions().len() / 3)
            .map(|i| model::ModelVertex {
                position: self.tri_positions(i),
                tex_coords: [self.tex_coords()[i * 2], 1.0 - self.tex_coords()[i * 2 + 1]],
                normal: self.tri_normals(i),
            })
            .collect::<Vec<_>>();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Index Buffer"),
            contents: bytemuck::cast_slice(&self.indices()),
            usage: wgpu::BufferUsages::INDEX,
        });

        Mesh {
            name: String::from("A Mesh"),
            vertex_buffer,
            index_buffer,
            num_elements: self.indices().len() as u32,
            material: material_id,
        }
    }
}
impl ProcessingModel for tobj::Model {
    fn positions(&self) -> Vec<f32> {
        self.mesh.positions.clone()
    }

    fn normals(&self) -> Vec<f32> {
        self.mesh.normals.clone()
    }

    fn indices(&self) -> Vec<u32> {
        self.mesh.indices.clone()
    }

    fn tex_coords(&self) -> Vec<f32> {
        self.mesh.texcoords.clone()
    }
}

pub trait ProcessingMaterial {
    fn name(&self) -> &str;
    async fn diffuse(
        &self,
        device: &wgpu::Device,
        queue: &Queue,
    ) -> Result<texture::Texture, Box<dyn Error>>;

    async fn read(
        self,
        device: &wgpu::Device,
        queue: &Queue,
        layout: &BindGroupLayout,
    ) -> Result<Material, Box<dyn Error>>
    where
        Self: std::marker::Sized,
    {
        let diffuse = self.diffuse(device, queue).await?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse.sampler),
                },
            ],
            label: None,
        });

        Ok(Material {
            name: self.name().to_string(),
            diffuse_texture: diffuse,
            bind_group,
        })
    }
}

impl ProcessingMaterial for tobj::Material {
    fn name(&self) -> &str {
        &self.name
    }

    async fn diffuse(&self, device: &Device, queue: &Queue) -> Result<Texture, Box<dyn Error>> {
        load_texture(
            self.diffuse_texture
                .clone()
                .ok_or(ModelLoadError::TextureDiffuseMissing)?,
            device,
            queue,
        )
        .await
    }
}

pub async fn load_string(file_name: &str) -> Result<String, Box<dyn Error>> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(file_name);
            let txt = reqwest::get(url)
                .await?
                .text()
                .await?;
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("res")
                .join(file_name);
            let txt = std::fs::read_to_string(path)?;
        }
    }

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(file_name);
            let data = reqwest::get(url)
                .await?
                .bytes()
                .await?
                .to_vec();
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("res")
                .join(file_name);
            let data = std::fs::read(path)?;
        }
    }

    Ok(data)
}

pub async fn load_texture(
    file_name: String,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<texture::Texture, Box<dyn Error + 'static>> {
    let data = load_binary(&file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, &file_name).map_err(|e| e.into())
}

#[derive(thiserror::Error, Debug)]
enum ModelLoadError {
    #[error("Model has no Diffuse Texture")]
    TextureDiffuseMissing,
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> Result<model::Model, Box<dyn std::error::Error + 'static>> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let materials = futures::future::join_all(
        obj_materials?
            .into_iter()
            .map(|m| m.read(device, queue, layout))
            .collect::<Vec<_>>(),
    )
    .await
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;

    let meshes = models
        .into_iter()
        .map(|m| m.mesh(device, m.mesh.material_id.unwrap_or(0)))
        .collect();

    Ok(model::Model { meshes, materials })
}
