use std::{
    fs::File,
    hash::{DefaultHasher, Hash, Hasher},
    path::Path,
};

use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use bincode::{Decode, Encode};
use glam::{ivec3, IVec3};
use itertools::Itertools;

use super::map::Block;

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

pub enum ChunkScramble {
    Normal,
    Inverse,
    Random,
}

#[derive(Copy, Clone, Encode, Decode)]
pub struct Chunk {
    pub blocks: Slice3,
}

impl Chunk {
    fn generate_normal(world_pos: IVec3) -> Chunk {
        let blocks = itertools::iproduct!(0..CHUNK_SIZE.0, 0..CHUNK_SIZE.1, 0..CHUNK_SIZE.2)
            .map(|(x, z, y)| {
                let tile_pos = ivec3(x as _, y as _, z as _);
                let tile_pos_worldspace = (tile_pos + (world_pos * CHUNK_SIZE.0 as i32)).as_vec3();

                let sines =
                    f32::sin(tile_pos_worldspace.x * 0.1) + f32::sin(tile_pos_worldspace.z * 0.1);

                // Pretty arbitrary numbers! Just trying to get something interesting
                let n = (((sines / 4. + 0.5) * CHUNK_SIZE.2 as f32).round() as i32)
                    <= tile_pos_worldspace.y as _;

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

    fn generate_callback(method: ChunkScramble) -> fn(IVec3) -> Chunk {
        use ChunkScramble as C;
        match method {
            C::Normal => Chunk::generate_normal,
            C::Inverse => |p| Chunk {
                blocks: Chunk::generate_normal(p)
                    .blocks
                    .iter()
                    .map(|b| match b {
                        Block::Air => Block::Brick,
                        Block::Brick => Block::Air,
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

            log::warn!("Wrote to file {file_name} with {encoded}b.");
        }

        // We are going to use LocalStorage for web. I don't like it either.
        #[cfg(target_arch = "wasm32")]
        {
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
                log::warn!("Load Chunk!");
                let mut file = File::open(file_path).unwrap();

                let decoded = bincode::decode_from_std_read(&mut file, config)?;

                Ok(Some(decoded))
            } else {
                log::warn!("Chunk not loaded!");
                Ok(None)
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
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
        match Chunk::load_from_file(map_pos)? {
            Some(chunk) => Ok(chunk),
            None => {
                let new_chunk = Chunk::generate(map_pos, ChunkScramble::Normal);
                new_chunk.save(map_pos)?;
                Ok(new_chunk)
            }
        }
    }
}

// Rust stdlib
fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}