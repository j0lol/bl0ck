use super::map::BlockKind;
use crate::ConnectionOnlyOnNative;
use bincode::{Decode, Encode};
use glam::{ivec3, IVec3};
use itertools::Itertools;
use std::sync::Mutex;
use std::{
    collections::HashMap,
    fs::File,
    hash::{DefaultHasher, Hash, Hasher},
    path::Path,
    sync::LazyLock,
};

#[cfg(not(target_arch = "wasm32"))]
static CHUNK_FILE_CACHE: LazyLock<Mutex<HashMap<IVec3, Chunk>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// I have arbitrarily decided that this is (x,z,y) where +y is up.
pub(crate) const CHUNK_SIZE: (usize, usize, usize) = (16, 16, 16);

// A [Block; X*Y*Z] would be a much more efficient datatype, but, well...
pub type Slice3 = [BlockKind; CHUNK_SIZE.0 * CHUNK_SIZE.1 * CHUNK_SIZE.2];

pub fn sl3get(sl3: &Slice3, x: usize, y: usize, z: usize) -> BlockKind {
    sl3[y + CHUNK_SIZE.2 * (z + CHUNK_SIZE.1 * x)]
}
pub fn sl3get_opt(sl3: &Slice3, x: usize, y: usize, z: usize) -> Option<BlockKind> {
    sl3.get(y + CHUNK_SIZE.2 * (z + CHUNK_SIZE.1 * x)).copied()
}
pub fn sl3set(sl3: &mut Slice3, x: usize, y: usize, z: usize, new: BlockKind) {
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
                    <= -tile_pos_worldspace.y as _;

                if n {
                    BlockKind::Brick
                } else {
                    BlockKind::Air
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

    pub fn save(
        &self,
        map_pos: IVec3,
        conn: &mut ConnectionOnlyOnNative,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = bincode::config::standard();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let encoded = bincode::encode_to_vec(self, config)?;

            let mut stmt = conn.prepare_cached(
                r#"
                INSERT INTO chunks (x,y,z,data)
                VALUES (?,?,?,?)
            "#,
            )?;
            stmt.insert((map_pos.x, map_pos.y, map_pos.z, encoded))?;
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
    fn load_from_file(
        map_pos: IVec3,
        conn: &mut ConnectionOnlyOnNative,
    ) -> Result<Option<Chunk>, Box<dyn std::error::Error>> {
        let config = bincode::config::standard();
        let file_hash = calculate_hash(&map_pos);
        let file_name = format!("chunk_{}.bl0ck", file_hash);

        #[cfg(not(target_arch = "wasm32"))]
        {
            // let file_path = Path::new("./save/chunk/").join(Path::new(&file_name));
            // if file_path.exists() {
            // log::warn!("Load Chunk!");
            // let mut file = File::open(file_path).unwrap();

            let mut stmt = conn.prepare_cached(
                r#"
                SELECT (data) from chunks
                WHERE (x,y,z) == (?,?,?)
            "#,
            )?;
            let i: Vec<u8> =
                match stmt.query_row((map_pos.x, map_pos.y, map_pos.z), |f| f.get("data")) {
                    Ok(x) => x,
                    Err(rusqlite::Error::QueryReturnedNoRows) => {
                        return Ok(None);
                    }
                    Err(e) => {
                        return Err(e.into());
                    }
                };

            let (decoded, _) = bincode::decode_from_slice(i.as_slice(), config)?;

            Ok(Some(decoded))
            // } else {
            //     log::warn!("Chunk not loaded!");
            //     Ok(None)
            // }
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

    pub fn load(
        map_pos: IVec3,
        conn: &mut ConnectionOnlyOnNative,
    ) -> Result<Chunk, Box<dyn std::error::Error>> {
        #[cfg(not(target_arch = "wasm32"))]
        let cached = CHUNK_FILE_CACHE.lock().unwrap().contains_key(&map_pos);
        #[cfg(target_arch = "wasm32")]
        let cached = false;

        if cached {
            // log::warn!("Cache hit!");
            #[cfg(not(target_arch = "wasm32"))]
            return Ok(CHUNK_FILE_CACHE.lock().unwrap()[&map_pos]);
            #[cfg(target_arch = "wasm32")]
            return unreachable!();
        } else {
            // log::warn!("Cache miss!");
            let chunk = match Chunk::load_from_file(map_pos, conn)? {
                Some(chunk) => chunk,
                None => {
                    let new_chunk = Chunk::generate(map_pos, ChunkScramble::Normal);
                    new_chunk.save(map_pos, conn)?;
                    new_chunk
                }
            };
            #[cfg(not(target_arch = "wasm32"))]
            CHUNK_FILE_CACHE.lock().unwrap().insert(map_pos, chunk);
            Ok(chunk)
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
            // let _ = Chunk::load(ivec3(x, y, z));
        }
    }
}
