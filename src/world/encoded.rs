use std::collections::HashMap;
use bincode::{Decode, Encode};
use rollgrid::rollgrid3d::RollGrid3D;

use super::{chunk::Chunk, map::WorldMap};


#[derive(Decode, Encode)]
struct EncodedWorldMap {
    chunks: HashMap<(i32, i32, i32), Chunk>,
    size: (u32, u32, u32),
    grid_offset: (i32, i32, i32),
}

impl EncodedWorldMap {
    fn to_rollgrid(self) -> WorldMap {
        WorldMap {
            chunks: RollGrid3D::new(self.size, self.grid_offset, |p| self.chunks[&p]),
        }
    }
}

impl bincode::Encode for WorldMap {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        let mut map: HashMap<(i32, i32, i32), Chunk> = HashMap::new();
        for ((x, y, z), chunk) in self.chunks.iter() {
            map.insert((x, y, z), *chunk);
        }
        bincode::Encode::encode(
            &EncodedWorldMap {
                chunks: map,
                size: self.chunks.size(),
                grid_offset: self.chunks.offset(),
            },
            encoder,
        )?;
        Ok(())
    }
}

impl bincode::Decode for WorldMap {
    fn decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        Ok(EncodedWorldMap::decode(decoder)?.to_rollgrid())
    }
}
impl<'de> bincode::BorrowDecode<'de> for WorldMap {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        Ok(EncodedWorldMap::borrow_decode(decoder)?.to_rollgrid())
    }
}
