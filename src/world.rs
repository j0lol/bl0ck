pub(crate) mod map;

use std::fs::File;

use bincode::{Decode, Encode};

use map::WorldMap;

#[derive(Encode, Decode)]
pub struct World {
    pub map: WorldMap,
}
impl World {
    pub fn save(&self) -> Result<(), bincode::error::EncodeError> {
        let config = bincode::config::standard();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut file = File::create("save.bl0ck").unwrap();
            let encoded = bincode::encode_into_std_write(self, &mut file, config)?;

            log::info!("{encoded} bytes written.");
        }
        #[cfg(target_arch = "wasm32")]
        unimplemented!();

        Ok(())
    }

    pub fn load() -> Result<World, bincode::error::DecodeError> {
        let config = bincode::config::standard();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut file = File::open("save.bl0ck").unwrap();

            let decoded = bincode::decode_from_std_read(&mut file, config)?;

            Ok(decoded)
        }
        #[cfg(target_arch = "wasm32")]
        unimplemented!();
    }
}
