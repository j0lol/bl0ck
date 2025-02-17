pub(crate) mod map;
pub(crate) mod encoded;
pub(crate) mod chunk;

use bincode::{Decode, Encode};

use glam::ivec3;
use map::WorldMap;

#[derive(Encode, Decode)]
pub struct World {
    pub map: WorldMap,
}
impl World {
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        for ((x,y,z), chunk) in self.map.chunks.iter() {
            chunk.save(ivec3(x,y,z))?;
        }
        Ok(())
    }
}
