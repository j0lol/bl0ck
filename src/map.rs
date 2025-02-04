use glam::{ivec2, IVec2};
use std::collections::HashMap;

// I have arbitrarily decided that this is (x,z,y) where +y is up.

pub(crate) const CHUNK_SIZE: (usize, usize, usize) = (16, 16, 16);

// A [Block; X*Y*Z] would be a much more efficient datatype, but, well...
pub type Slice3 = [[[Block; CHUNK_SIZE.0]; CHUNK_SIZE.1]; CHUNK_SIZE.2];

pub fn sl3get(sl3: &Slice3, x: usize, y: usize, z: usize) -> Block {
    sl3[y][z][x]
}
pub fn sl3set(sl3: &mut Slice3, x: usize, y: usize, z: usize, new: Block) {
    sl3[y][z][x] = new;
}

pub struct WorldMap {
    pub chunks: HashMap<IVec2, Chunk>,
}

pub struct Chunk {
    pub blocks: Slice3,
}

#[derive(Copy, Clone, Default)]
#[repr(u32)]
pub enum Block {
    #[default]
    Air = 0,
    Brick,
}

fn new_chunk(world_x: i32, world_z: i32) -> Chunk {
    let mut blocks = Slice3::default();

    for (x, z, y) in itertools::iproduct!(0..CHUNK_SIZE.0, 0..CHUNK_SIZE.1, 0..CHUNK_SIZE.2) {
        let (xf, zf) = (
            (x as i32 + (world_x * CHUNK_SIZE.0 as i32)) as f32,
            (z as i32 + (world_z * CHUNK_SIZE.0 as i32)) as f32,
        );

        let sines = f32::sin(xf * 0.1) + f32::sin(zf * 0.1);

        // Pretty arbitrary numbers! Just trying to get something interesting
        let n = (((sines / 4. + 0.5) * CHUNK_SIZE.2 as f32).round() as i32) <= y as _;
        sl3set(&mut blocks, x, y, z, {
            if n { Block::Brick } else { Block::Air }
        });
    }

    Chunk { blocks }
}

pub fn new_map() -> WorldMap {

    const INITIAL_GENERATION_SIZE: usize = 11;
    let iter = (-(INITIAL_GENERATION_SIZE as i32/2)..).take(INITIAL_GENERATION_SIZE);

    let mut chunks = HashMap::new();

    for (x, z) in itertools::iproduct!(iter.clone(), iter) {
        chunks.insert(ivec2(x,z) ,new_chunk(x, z));
    }

    WorldMap { chunks }
}