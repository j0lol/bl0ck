use super::map::BlockKind;
use crate::gfx::model::{Material, Mesh, Model, ModelVertex};
use crate::gfx::primitive::cube::Faces;
use crate::gfx::primitive::PrimitiveMeshBuilder;
use crate::gfx::resources::ProcessingModel;
use bincode::{Decode, Encode};
use futures::StreamExt;
use glam::{ivec3, vec3, I8Vec3, IVec3, Vec2, Vec3};
use itertools::{iproduct, Itertools};
use std::sync::{Arc, Mutex};
use std::{
    collections::HashMap,
    fs::File,
    hash::{DefaultHasher, Hash, Hasher},
    path::Path,
    sync::LazyLock,
};
use wgpu::util::DeviceExt;
use wgpu::Device;

#[cfg(not(target_arch = "wasm32"))]
static CHUNK_FILE_CACHE: LazyLock<Mutex<HashMap<IVec3, Chunk>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// I have arbitrarily decided that this is (front,z,y) where +y is up.
pub(crate) const CHUNK_SIZE: (usize, usize, usize) = (16, 16, 16);

// A [Block; X*Y*Z] would be a much more efficient datatype, but, well...
pub type Slice3 = [BlockKind; CHUNK_SIZE.0 * CHUNK_SIZE.1 * CHUNK_SIZE.2];

pub const FULL_CHUNK: Chunk = Chunk {
    blocks: [BlockKind::Brick; CHUNK_SIZE.0 * CHUNK_SIZE.1 * CHUNK_SIZE.2],
};
pub const HALF_CHUNK: fn() -> Chunk = || {
    let mut blocks = [BlockKind::Air; CHUNK_SIZE.0 * CHUNK_SIZE.1 * CHUNK_SIZE.2];

    let block = |b| if b { BlockKind::Brick } else { BlockKind::Air };
    for (x, y, z) in iproduct!(0..CHUNK_SIZE.0, 10..CHUNK_SIZE.1, 0..CHUNK_SIZE.2) {
        blocks[Chunk::linearize(x, y, z).unwrap()] =
            block(f32::sin(x as f32) + f32::sin(z as f32) > 0.7);
    }
    Chunk { blocks }
};

pub const TEST_CHUNK: fn() -> Chunk = || {
    let mut blocks = [BlockKind::Air; CHUNK_SIZE.0 * CHUNK_SIZE.1 * CHUNK_SIZE.2];

    // blocks[Chunk::linearize(10, 10, 10)] = BlockKind::Brick;
    // blocks[Chunk::linearize(6, 6, 6)] = BlockKind::Brick;
    // blocks[Chunk::linearize(6, 7, 6)] = BlockKind::Brick;
    //
    // blocks[Chunk::linearize(4, 2, 2)] = BlockKind::Brick;
    // blocks[Chunk::linearize(3, 2, 2)] = BlockKind::Brick;
    // blocks[Chunk::linearize(2, 2, 2)] = BlockKind::Brick;
    //
    // blocks[Chunk::linearize(2, 2, 8)] = BlockKind::Brick;
    // blocks[Chunk::linearize(2, 2, 9)] = BlockKind::Brick;
    // blocks[Chunk::linearize(2, 2, 10)] = BlockKind::Brick;
    // blocks[Chunk::linearize(2, 2, 11)] = BlockKind::Brick;

    Chunk { blocks }
};

pub enum ChunkScramble {
    Normal,
    Inverse,
    Random,
}

pub trait ChunkTrait {
    type Node;

    const X: usize;
    const Y: usize;
    const Z: usize;

    fn size() -> usize {
        Self::X * Self::Y * Self::Z
    }

    fn linearize(x: usize, y: usize, z: usize) -> Option<usize> {
        if (x > Self::X) || (y > Self::Y) || (z > Self::Z) {
            None
        } else {
            Some(x + (y * Self::X) + (z * Self::X * Self::Y))
        }
    }

    fn delinearize(mut index: usize) -> (usize, usize, usize) {
        let z = index / (Self::X * Self::Y);
        index -= z * (Self::X * Self::Y);

        let y = index / Self::X;
        index -= y * Self::X;

        let x = index;

        (x, y, z)
    }

    fn get(&self, x: usize, y: usize, z: usize) -> Self::Node;
    fn get_opt(&self, x: usize, y: usize, z: usize) -> Option<Self::Node>;
}

pub trait Voxel: Eq {
    fn visibility(&self) -> bool;
}

#[derive(Copy, Clone, Debug)]
pub struct Quad {
    pub voxel: [usize; 3],
    pub width: u32,
    pub height: u32,
}

#[derive(Default, Debug)]
pub struct QuadGroups {
    pub groups: [Vec<Quad>; 6],
}

pub struct Face<'a> {
    pub(crate) side: Vec3,
    pub(crate) quad: &'a Quad,
}

impl<'a> Face<'a> {
    pub fn indices(&self, start: u32) -> [u32; 6] {
        [start, start + 2, start + 1, start + 1, start + 2, start + 3]
    }

    pub fn positions(&self, voxel_size: f32) -> [Vec3; 4] {
        let positions = match self.side {
            Vec3::NEG_X => [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            Vec3::X => [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            Vec3::NEG_Y => [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            Vec3::Y => [
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            Vec3::NEG_Z => [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            Vec3::Z => [
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            _ => unreachable!(),
        };

        let (x, y, z) = (
            (self.quad.voxel[0] - 1) as f32,
            (self.quad.voxel[1] - 1) as f32,
            (self.quad.voxel[2] - 1) as f32,
        );

        [
            [
                x * voxel_size + positions[0][0] * voxel_size,
                y * voxel_size + positions[0][1] * voxel_size,
                z * voxel_size + positions[0][2] * voxel_size,
            ]
            .into(),
            [
                x * voxel_size + positions[1][0] * voxel_size,
                y * voxel_size + positions[1][1] * voxel_size,
                z * voxel_size + positions[1][2] * voxel_size,
            ]
            .into(),
            [
                x * voxel_size + positions[2][0] * voxel_size,
                y * voxel_size + positions[2][1] * voxel_size,
                z * voxel_size + positions[2][2] * voxel_size,
            ]
            .into(),
            [
                x * voxel_size + positions[3][0] * voxel_size,
                y * voxel_size + positions[3][1] * voxel_size,
                z * voxel_size + positions[3][2] * voxel_size,
            ]
            .into(),
        ]
    }

    pub fn normals(&self) -> [Vec3; 4] {
        [self.side; 4]
    }

    pub fn uvs(&self, flip_u: bool, flip_v: bool) -> [Vec2; 4] {
        match (flip_u, flip_v) {
            (true, true) => [Vec2::ONE, Vec2::Y, Vec2::X, Vec2::ZERO],
            (true, false) => [Vec2::X, Vec2::ZERO, Vec2::ONE, Vec2::Y],
            (false, true) => [Vec2::Y, Vec2::ONE, Vec2::ZERO, Vec2::X],
            (false, false) => [Vec2::ZERO, Vec2::X, Vec2::Y, Vec2::ONE],
        }
    }

    pub fn voxel(&self) -> [usize; 3] {
        self.quad.voxel
    }
}

pub fn side_lookup(value: usize) -> Vec3 {
    match value {
        0 => Vec3::NEG_X,
        1 => Vec3::X,
        2 => Vec3::NEG_Y,
        3 => Vec3::Y,
        4 => Vec3::NEG_Z,
        5 => Vec3::Z,
        _ => unreachable!(),
    }
}

impl QuadGroups {
    pub fn iter(&self) -> impl Iterator<Item = Face> {
        self.groups
            .iter()
            .enumerate()
            .flat_map(|(index, quads)| quads.iter().map(move |quad| (index, quad)))
            .map(|(index, quad)| Face {
                side: side_lookup(index),
                quad,
            })
    }
}

#[derive(Copy, Clone, Encode, Decode)]
pub struct Chunk {
    pub blocks: Slice3,
}

impl ChunkTrait for Chunk {
    type Node = BlockKind;
    const X: usize = 16;
    const Y: usize = 16;
    const Z: usize = 16;

    fn get(&self, x: usize, y: usize, z: usize) -> Self::Node {
        self.blocks[Chunk::linearize(x, y, z).unwrap()]
    }
    fn get_opt(&self, x: usize, y: usize, z: usize) -> Option<Self::Node> {
        self.blocks.get(Chunk::linearize(x, y, z)?).copied()
    }
}

pub fn generate_mesh<C, T>(chunk: &C) -> QuadGroups
where
    C: ChunkTrait<Node = T>,
    T: Voxel,
{
    todo!()
}

pub fn simple_mesh<C, T>(chunk: &C) -> QuadGroups
where
    C: ChunkTrait<Node = T>,
    T: Voxel,
{
    assert!(C::X >= 2);
    assert!(C::Y >= 2);
    assert!(C::Z >= 2);

    let mut buffer = QuadGroups::default();

    for i in 0..C::size() {
        let (x, y, z) = C::delinearize(i);

        if (x > 0 && x < C::X - 1) && (y > 0 && y < C::Y - 1) && (z > 0 && z < C::Z - 1) {
            let voxel = chunk.get(x, y, z);

            if voxel.visibility() {
                let neighbors = [
                    chunk.get(x - 1, y, z),
                    chunk.get(x + 1, y, z),
                    chunk.get(x, y - 1, z),
                    chunk.get(x, y + 1, z),
                    chunk.get(x, y, z - 1),
                    chunk.get(x, y, z + 1),
                ];

                for (i, neighbor) in neighbors.into_iter().enumerate() {
                    let other = neighbor.visibility();

                    let generate = !other;

                    if generate {
                        buffer.groups[i].push(Quad {
                            voxel: [x, y, z],
                            width: 1,
                            height: 1,
                        });
                    }
                }
            } else {
                continue;
            }
        }
    }

    buffer
}

impl Voxel for BlockKind {
    fn visibility(&self) -> bool {
        *self == BlockKind::Brick
    }
}

impl Chunk {
    fn generate_normal(world_pos: IVec3) -> Chunk {
        let mut blocks = vec![];
        for (x, y, z) in iproduct!(0..CHUNK_SIZE.0, 0..CHUNK_SIZE.1, 0..CHUNK_SIZE.2) {
            let tile_pos = ivec3(x as _, y as _, z as _);
            let tile_pos_worldspace = (tile_pos + (world_pos * CHUNK_SIZE.0 as i32)).as_vec3();

            // let sines =
            //     f32::sin(tile_pos_worldspace.x * 0.1) + f32::sin(tile_pos_worldspace.z * 0.1);
            //
            // // Pretty arbitrary numbers! Just trying to get something interesting
            // let n = (((sines / 4. + 0.5) * CHUNK_SIZE.2 as f32).round() as i32)
            //     <= -tile_pos_worldspace.y as _;

            let n = (tile_pos_worldspace.y < 10.0);

            blocks.push(if n { BlockKind::Brick } else { BlockKind::Air });
        }

        Chunk {
            blocks: blocks.into_iter().collect_array().unwrap(),
        }
    }

    fn generate_callback(method: ChunkScramble) -> fn(IVec3) -> Chunk {
        use ChunkScramble as C;
        match method {
            C::Normal => Chunk::generate_normal,
            C::Inverse => |p| Chunk {
                blocks: Chunk::generate_normal(p)
                    .blocks
                    .iter()
                    .map(|b| match b {
                        BlockKind::Air => BlockKind::Brick,
                        BlockKind::Brick => BlockKind::Air,
                    })
                    .collect_array()
                    .unwrap(),
            },
            C::Random => |_| Chunk {
                #[cfg(not(target_arch = "wasm32"))]
                blocks: {
                    use rand::Rng;
                    rand::rng().random()
                },
                #[cfg(target_arch = "wasm32")]
                blocks: { panic!("i hate the web") },
            },
        }
    }
    pub fn generate(map_pos: IVec3, method: ChunkScramble) -> Chunk {
        Chunk::generate_callback(method)(map_pos)
    }

    pub fn save(&self, map_pos: IVec3) -> Result<(), Box<dyn std::error::Error>> {
        let config = bincode::config::standard();

        let file_hash = calculate_hash(&map_pos);
        let file_name = format!("chunk_{}.bl0ck", file_hash);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let file_path = Path::new("./save/chunk/").join(Path::new(&file_name));
            let mut file = File::create(&file_path).unwrap();
            let encoded = bincode::encode_into_std_write(self, &mut file, config)?;

            log::info!("Wrote to file {file_name} with {encoded}b.");
        }

        // We are going to use LocalStorage for web. I don't like it either.
        #[cfg(target_arch = "wasm32")]
        {
            use base64::prelude::{Engine, BASE64_STANDARD};
            let encoded = bincode::encode_to_vec(self, config)?;
            let encoded = BASE64_STANDARD.encode(encoded);

            let store = web_sys::window().unwrap().local_storage().unwrap().unwrap();
            store.set(&file_name, &encoded);
        }
        Ok(())
    }
    fn load_from_file(map_pos: IVec3) -> Result<Option<Chunk>, Box<dyn std::error::Error>> {
        let config = bincode::config::standard();
        let file_hash = calculate_hash(&map_pos);
        let file_name = format!("chunk_{}.bl0ck", file_hash);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let file_path = Path::new("./save/chunk/").join(Path::new(&file_name));
            if file_path.exists() {
                log::info!("Load Chunk!");
                let mut file = File::open(file_path).unwrap();

                let decoded = bincode::decode_from_std_read(&mut file, config)?;

                Ok(Some(decoded))
            } else {
                log::info!("Chunk not loaded!");
                Ok(None)
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            use base64::prelude::{Engine, BASE64_STANDARD};
            let store = web_sys::window().unwrap().local_storage().unwrap().unwrap();
            if let Ok(Some(s)) = store.get(&file_name) {
                let s = BASE64_STANDARD.decode(s)?;
                let (decoded, _) = bincode::decode_from_slice(&s[..], config)?;

                Ok(Some(decoded))
            } else {
                Ok(None)
            }
        }
    }

    pub fn load(map_pos: IVec3) -> Result<Chunk, Box<dyn std::error::Error>> {
        // return Ok(Chunk::generate(map_pos, ChunkScramble::Normal));
        // return Ok(HALF_CHUNK());

        #[cfg(not(target_arch = "wasm32"))]
        let cached = CHUNK_FILE_CACHE.lock().unwrap().contains_key(&map_pos);
        #[cfg(target_arch = "wasm32")]
        let cached = false;

        if cached {
            log::info!("Cache hit!");
            #[cfg(not(target_arch = "wasm32"))]
            return Ok(CHUNK_FILE_CACHE.lock().unwrap()[&map_pos]);
            #[cfg(target_arch = "wasm32")]
            return unreachable!();
        } else {
            log::info!("Cache miss!");
            let chunk = match Chunk::load_from_file(map_pos)? {
                Some(chunk) => chunk,
                None => {
                    let new_chunk = Chunk::generate(map_pos, ChunkScramble::Normal);
                    new_chunk.save(map_pos)?;
                    new_chunk
                }
            };
            #[cfg(not(target_arch = "wasm32"))]
            CHUNK_FILE_CACHE.lock().unwrap().insert(map_pos, chunk);
            Ok(chunk)
        }
    }

    fn build_mesh(&self, device: &wgpu::Device) -> Result<Mesh, Box<dyn std::error::Error>> {
        let quad_groups = simple_mesh(self);

        Ok(quad_groups.mesh(device, 0))
    }

    pub async fn model(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tex_bgl: &wgpu::BindGroupLayout,
    ) -> Result<Model, Box<dyn std::error::Error>> {
        let mesh = simple_mesh(self);
        let vertices = mesh.positions();
        let indices = mesh.indices();

        let mut vertices = vec![];
        for face in mesh.iter() {
            let vertex = ModelVertex {
                position: face.side.into(),
                normal: face.normals()[0].into(),
                tex_coords: face.uvs(false, true)[0].into(),
            };
            vertices.push(vertex);
        }

        Ok(crate::gfx::primitive::PrimitiveMesh::new(
            device, &queue, &tex_bgl, &*vertices, &indices,
        )
        .await
        .model)
        // Ok(Model {
        //     meshes: vec![self.build_mesh(device)?],
        //     materials: crate::gfx::resources::load_model(
        //         "blender_default_cube.obj",
        //         &device,
        //         &queue,
        //         &tex_bgl,
        //     )
        //     .await?
        //     .materials,
        // })
    }

    pub fn primitive_model(
        &self,
        (cx, cy, cz): (i32, i32, i32),
        device: &Device,
        material: &Arc<Vec<Material>>,
    ) -> Option<Model> {
        let mut mesh = PrimitiveMeshBuilder::new();

        for (x, y, z) in iproduct!(0..Chunk::X, 0..Chunk::Y, 0..Chunk::Z) {
            match self.get(x, y, z) {
                BlockKind::Air => continue,
                BlockKind::Brick => {
                    let faces = Faces::normals().map(
                        |IVec3 {
                             x: nx,
                             y: ny,
                             z: nz,
                         }| {
                            match self.get_opt(
                                (x as i32 + nx) as usize,
                                (y as i32 + ny) as usize,
                                (z as i32 + nz) as usize,
                            ) {
                                None => {
                                    //dbg!("oob!");
                                    true
                                }
                                Some(BlockKind::Brick) => false,
                                Some(BlockKind::Air) => true,
                            }
                        },
                    );
                    let faces = Faces::from_arr(faces);

                    let offset = |x: usize, cx: i32, max: usize| {
                        (cx as f32 * 2.0 * max as f32) + x as f32 * 2.
                    };

                    mesh = mesh.cube(
                        faces,
                        offset(x, cx, Chunk::X),
                        offset(y, cy, Chunk::Y),
                        offset(z, cz, Chunk::Z),
                    );
                }
            }
        }
        if let Some(mesh) = mesh.build(device, material) {
            Some(mesh.model)
        } else {
            None
        }
    }
}

// Rust stdlib
fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

pub fn preload_chunk_cache() {
    #[cfg(target_arch = "wasm32")]
    return;

    #[cfg(not(target_arch = "wasm32"))]
    {
        // range
        let r: i32 = 8; // normally 8 or so
        let _3diter = itertools::iproduct!(-r..r, -r..r, -r..r);

        for (x, y, z) in _3diter {
            let _ = Chunk::load(ivec3(x, y, z));
        }
    }
}

impl ProcessingModel for QuadGroups {
    fn positions(&self) -> Vec<f32> {
        self.iter()
            .flat_map(|f| f.positions(1.0).map(|Vec3 { x, y, z }| vec![x, y, z]))
            .flatten()
            .collect_vec()
    }

    fn normals(&self) -> Vec<f32> {
        self.iter()
            .flat_map(|f| f.normals().map(|Vec3 { x, y, z }| vec![x, y, z]))
            .flatten()
            .collect_vec()
    }

    fn indices(&self) -> Vec<u32> {
        self.iter()
            .flat_map(|f| f.indices(self.positions().len() as u32))
            .collect_vec()
    }

    fn tex_coords(&self) -> Vec<f32> {
        self.iter()
            .flat_map(|f| f.uvs(false, true).map(|Vec2 { x, y }| vec![x, y]))
            .flatten()
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_chunk_meshing() {
        // Handle to our GPU backends
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,

            // WebGPU is sadly not supported in most browsers (yet!)
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,

            ..Default::default()
        });

        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    // lo-power or hi-perf
                    power_preference: wgpu::PowerPreference::default(),
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .unwrap()
        });

        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        required_features: if adapter.get_info().backend == wgpu::Backend::Gl {
                            wgpu::Features::empty()
                        } else {
                            wgpu::Features::POLYGON_MODE_LINE
                        },

                        // WebGL does not support all of wgpu's features
                        required_limits: if adapter.get_info().backend == wgpu::Backend::Gl {
                            wgpu::Limits::downlevel_webgl2_defaults()
                        } else {
                            wgpu::Limits::default()
                        },
                        label: None,
                        memory_hints: Default::default(),
                    },
                    None,
                )
                .await
                .unwrap()
        });

        let chunk = Chunk::generate_normal((0, 0, 0).into());

        // for i in 0..Chunk::size() {
        //     let (front, y, z) = Chunk::delinearize(i);
        //
        //     let voxel = if ((front * front + y * y + z * z) as f32).sqrt() < 8.0 {
        //         if y > 4 {
        //             MyVoxel::Opaque(1)
        //         } else if front > 4 {
        //             MyVoxel::Transparent(1)
        //         } else {
        //             MyVoxel::Opaque(2)
        //         }
        //     } else {
        //         MyVoxel::Empty
        //     };
        //
        //     chunk.voxels[i] = voxel;
        // }

        let result = simple_mesh(&chunk);

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs = Vec::new();

        for face in result.iter() {
            positions.extend_from_slice(&face.positions(1.0).map(|x| x.into())); // Voxel size is 1m
            indices.extend_from_slice(&face.indices(positions.len() as u32));
            normals.extend_from_slice(&face.normals().map(|x| x.into()));
            uvs.extend_from_slice(&face.uvs(false, true));
        }

        /*        let vertex = ModelVertex {
            positions: positions,
            normals: normals.as_slice(),
        }*/

        // let vertices = (0..m.mesh.positions.len() / 3)
        //     .map(|i| crate::gfx::model::ModelVertex {
        //         position: positions,
        //         tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
        //         normal: normals,
        //     })
        //     .collect::<Vec<_>>();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let _ = crate::gfx::model::Mesh {
            name: "".to_string(),
            vertex_buffer,
            index_buffer,
            num_elements: indices.len() as u32,
            material: 0,
        };

        // let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        //
        // mesh.set_indices(Some(Indices::U32(indices)));
        //
        // mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        // mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        // mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }

    #[test]
    fn chunk_get() {
        let chunk = HALF_CHUNK();

        assert_eq!(chunk.get(0, 0, 0), BlockKind::Air);

        dbg!(simple_mesh(&chunk));
    }
}
