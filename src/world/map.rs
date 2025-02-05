use bincode::{Decode, Encode};
use glam::{ivec2, ivec3, IVec2, IVec3};
use itertools::Itertools;
use rand::{distr::{Distribution, StandardUniform}, Rng};
use std::collections::HashMap;

// I have arbitrarily decided that this is (x,z,y) where +y is up.

pub(crate) const CHUNK_SIZE: (usize, usize, usize) = (16, 16, 16);

// A [Block; X*Y*Z] would be a much more efficient datatype, but, well...
pub type Slice3 = [Block; CHUNK_SIZE.0 * CHUNK_SIZE.1 * CHUNK_SIZE.2];

pub fn sl3get(sl3: &Slice3, x: usize, y: usize, z: usize) -> Block {
    sl3[y + CHUNK_SIZE.2 * (z + CHUNK_SIZE.1 * x)]
}
pub fn sl3set(sl3: &mut Slice3, x: usize, y: usize, z: usize, new: Block) {
    sl3[y + CHUNK_SIZE.2 * (z + CHUNK_SIZE.1 * x)] = new;
}

#[derive(Encode, Decode)]
pub struct WorldMap {
    pub chunks: HashMap<(i32, i32, i32), Chunk>,
}

#[derive(Encode, Decode)]
pub struct Chunk {
    pub blocks: Slice3,
}

#[derive(Copy, Clone, Default, Encode, Decode)]
#[repr(u32)]
pub enum Block {
    #[default]
    Air = 0,
    Brick,
}

impl Distribution<Block> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Block {
        match rng.random_range(0..=1) {
            0 => Block::Air,
            _ => Block::Brick,
        }
    }
}


fn new_chunk(world_pos: IVec3) -> Chunk {
    let blocks = itertools::iproduct!(0..CHUNK_SIZE.0, 0..CHUNK_SIZE.1, 0..CHUNK_SIZE.2)
        .map(|(x, z, y)| {
            let tile_pos = ivec3(x as _,y as _,z as _);

            // let (xf, zf, yf) = (
            //     (x as i32 + (world_x * CHUNK_SIZE.0 as i32)) as f32,
            //     (z as i32 + (world_z * CHUNK_SIZE.0 as i32)) as f32,
            //     (y as i32 + (world_y * CHUNK_SIZE.0 as i32)) as f32,
            // );
            let tile_pos_worldspace = (tile_pos + (world_pos * CHUNK_SIZE.0 as i32)).as_vec3();

            let sines = f32::sin(tile_pos_worldspace.x * 0.1) + f32::sin(tile_pos_worldspace.z * 0.1);

            // Pretty arbitrary numbers! Just trying to get something interesting
            let n = (((sines / 4. + 0.5) * CHUNK_SIZE.2 as f32).round() as i32) <= tile_pos_worldspace.y as _;

            if n {
                Block::Brick
            } else {
                Block::Air
            }
        })
        .collect_array()
        .unwrap();

    Chunk { blocks }
}

pub fn new_map() -> WorldMap {
    const INITIAL_GENERATION_SIZE: usize = 11;
    let iter = (-(INITIAL_GENERATION_SIZE as i32 / 2)..).take(INITIAL_GENERATION_SIZE);

    let mut chunks = HashMap::new();

    for (x, z) in itertools::iproduct!(iter.clone(), iter) {
        let p = ivec3(x,0,z);
        chunks.insert(p.into(), new_chunk(p));
        chunks.insert(p.with_y(1).into(), new_chunk(p.with_y(1)));

    }

    WorldMap { chunks }
}

pub fn chunk_scramble_dispatch(method: ChunkScramble) -> fn(IVec3) -> Chunk {
    use ChunkScramble as C;
    match method {
        C::Normal => new_chunk,
        C::Inverse => |p| Chunk {
            blocks: new_chunk(p)
                .blocks
                .iter()
                .map(|b| match b {
                    Block::Air => Block::Brick,
                    Block::Brick => Block::Air,
                })
                .collect_array()
                .unwrap(),
        },
        C::Random => |_| {
            Chunk {
                blocks: rand::rng().random()
            }
        }
    }
}

pub enum ChunkScramble {
    Normal,
    Inverse,
    Random,
}