pub(crate) mod chunk;
pub(crate) mod encoded;
pub(crate) mod map;

use bincode::{Decode, Encode};

use crate::ConnectionOnlyOnNative;
use glam::ivec3;
use map::WorldMap;

#[derive(Encode, Decode)]
pub struct World {
    pub map: WorldMap,
    pub remake: bool,
}
impl World {
    pub fn save(
        &self,
        conn: &mut ConnectionOnlyOnNative,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for ((x, y, z), chunk) in self.map.chunks.iter() {
            chunk.save(ivec3(x, y, z), conn)?;
        }
        Ok(())
    }
}
