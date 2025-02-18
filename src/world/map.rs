use super::chunk::Chunk;
use bincode::{Decode, Encode};
use glam::ivec3;
#[cfg(not(target_arch = "wasm32"))]
use rand::{
    distr::{Distribution, StandardUniform},
    Rng,
};
use rollgrid::rollgrid3d::RollGrid3D;

#[derive(Copy, Clone, Default, Encode, Decode, PartialEq, Eq)]
#[repr(u32)]
#[derive(Debug)]
pub enum BlockKind {
    #[default]
    Air = 0,
    Brick,
}

#[cfg(not(target_arch = "wasm32"))]
impl Distribution<BlockKind> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BlockKind {
        match rng.random_range(0..=1) {
            0 => BlockKind::Air,
            _ => BlockKind::Brick,
        }
    }
}

pub struct WorldMap {
    pub chunks: RollGrid3D<Chunk>,
}
pub(crate) const RENDER_GRID_SIZE: usize = 3;
pub fn new() -> WorldMap {
    WorldMap {
        chunks: RollGrid3D::new(
            (
                RENDER_GRID_SIZE as _,
                RENDER_GRID_SIZE as _,
                RENDER_GRID_SIZE as _,
            ),
            (0, 0, 0),
            |(x, y, z)| Chunk::load(ivec3(x, y, z)).unwrap(),
        ),
    }
}
