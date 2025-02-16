use bincode::{Decode, Encode};
use glam::ivec3;
#[cfg(not(target_arch = "wasm32"))]
use rand::{
    distr::{Distribution, StandardUniform},
    Rng,
};
use rollgrid::rollgrid3d::RollGrid3D;
use super::chunk::Chunk;

#[derive(Copy, Clone, Default, Encode, Decode)]
#[repr(u32)]
pub enum Block {
    #[default]
    Air = 0,
    Brick,
}

#[cfg(not(target_arch = "wasm32"))]
impl Distribution<Block> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Block {
        match rng.random_range(0..=1) {
            0 => Block::Air,
            _ => Block::Brick,
        }
    }
}

pub struct WorldMap {
    pub chunks: RollGrid3D<Chunk>,
}
pub fn new() -> WorldMap {
    const INITIAL_GENERATION_SIZE: usize = 5;

    WorldMap {
        chunks: RollGrid3D::new(
            (
                INITIAL_GENERATION_SIZE as _,
                2,
                INITIAL_GENERATION_SIZE as _,
            ),
            (0, 0, 0),
            |(x, y, z)| Chunk::load(ivec3(x, y, z)).unwrap(),
        ),
    }
}




