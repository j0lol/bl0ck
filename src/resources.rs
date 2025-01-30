use crate::{model, texture};
use cfg_if::cfg_if;
use std::error::Error;
use std::io::{BufReader, Cursor};
use wgpu::util::DeviceExt;

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let mut origin = location.origin().unwrap();
    if !origin.ends_with("res") {
        origin = format!("{}/res", origin);
    }
    let base = reqwest::Url::parse(&format!("{}/", origin)).unwrap();
    base.join(file_name).unwrap()
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
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<texture::Texture, Box<dyn Error + 'static>> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name).map_err(|e| e.into())
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

    let read_material = |material: tobj::Material| async {
        let diffuse_texture = load_texture(
            &material
                .diffuse_texture
                .ok_or(ModelLoadError::TextureDiffuseMissing)?,
            device,
            queue,
        )
        .await?;
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

        Ok::<_, Box<dyn Error>>(model::Material {
            name: material.name,
            diffuse_texture,
            bind_group,
        })
    };

    let mesh_positions = |m: &tobj::Model, i: usize| {
        [
            m.mesh.positions[i * 3],
            m.mesh.positions[i * 3 + 1],
            m.mesh.positions[i * 3 + 2],
        ]
    };
    let mesh_normals = |m: &tobj::Model, i: usize| {
        if m.mesh.normals.is_empty() {
            glam::Vec3::ZERO.into()
        } else {
            [
                m.mesh.normals[i * 3],
                m.mesh.normals[i * 3 + 1],
                m.mesh.normals[i * 3 + 2],
            ]
        }
    };

    let read_mesh = |m: tobj::Model| {
        let vertices = (0..m.mesh.positions.len() / 3)
            .map(|i| model::ModelVertex {
                position: mesh_positions(&m, i),
                tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
                normal: mesh_normals(&m, i),
            })
            .collect::<Vec<_>>();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{file_name:?} Vertex Buffer")),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{file_name:?} Index Buffer")),
            contents: bytemuck::cast_slice(&m.mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        model::Mesh {
            name: file_name.to_string(),
            vertex_buffer,
            index_buffer,
            num_elements: m.mesh.indices.len() as u32,
            material: m.mesh.material_id.unwrap_or(0),
        }
    };

    let materials = futures::future::join_all(
        obj_materials?
            .into_iter()
            .map(read_material)
            .collect::<Vec<_>>(),
    )
    .await
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;

    let meshes = models.into_iter().map(read_mesh).collect();

    Ok(model::Model { meshes, materials })
}