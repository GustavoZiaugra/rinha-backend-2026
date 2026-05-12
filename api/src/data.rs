use flate2::read::GzDecoder;
use std::io::Read;
use std::sync::OnceLock;

pub struct Dataset {
    pub centroids: Vec<f32>, // SOA: centroids[d*k + c]
    pub offsets: Vec<u32>,   // [k+1], cluster boundaries
    pub labels: Vec<u8>,     // padded to 8-byte per cluster
    pub blocks: Vec<i16>,    // cluster-major blocks, each block = [dims][8] i16
    pub k: usize,
    pub n: usize,
    pub padded_n: usize,
}

static DATASET: OnceLock<Dataset> = OnceLock::new();

pub fn init() {
    let ds = Dataset::load_embedded().expect("load IVF index");
    if DATASET.set(ds).is_err() {
        panic!("dataset already initialized");
    }
}

pub fn dataset() -> &'static Dataset {
    DATASET.get().expect("dataset not initialized")
}

impl Dataset {
    fn load_embedded() -> Result<Self, Box<dyn std::error::Error>> {
        static INDEX_GZ: &[u8] = include_bytes!("../../data/index.bin.gz");
        let mut gz = GzDecoder::new(&INDEX_GZ[..]);

        let mut magic = [0u8; 4];
        gz.read_exact(&mut magic)?;
        if &magic != b"IVF1" {
            return Err("bad magic".into());
        }

        let n = read_u32(&mut gz)? as usize;
        let k = read_u32(&mut gz)? as usize;
        let d = read_u32(&mut gz)? as usize;
        if d != 14 {
            return Err("expected d=14".into());
        }

        // Read centroids SOA f32 [d * k]
        let mut centroids = vec![0f32; d * k];
        let bytes = bytemuck::cast_slice_mut(&mut centroids);
        gz.read_exact(bytes)?;

        // Read offsets [k+1]
        let mut offsets = vec![0u32; k + 1];
        let off_bytes = bytemuck::cast_slice_mut(&mut offsets);
        gz.read_exact(off_bytes)?;

        let padded_n = offsets[k] as usize;

        // Read labels [padded_n]
        let mut labels = vec![0u8; padded_n];
        gz.read_exact(&mut labels)?;

        // Read blocks: total_blocks = padded_n / 8
        let total_blocks = padded_n / 8;
        let mut blocks = vec![0i16; total_blocks * d * 8];
        let block_bytes = bytemuck::cast_slice_mut(&mut blocks);
        gz.read_exact(block_bytes)?;

        Ok(Dataset {
            centroids,
            offsets,
            labels,
            blocks,
            k,
            n,
            padded_n,
        })
    }
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

// bytemuck helper - safe transmute for primitive arrays
pub mod bytemuck {
    pub fn cast_slice_mut<T, U>(s: &mut [T]) -> &mut [U] {
        let len = std::mem::size_of_val(s);
        let u_len = std::mem::size_of::<U>();
        assert!(len % u_len == 0);
        let new_len = len / u_len;
        unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut U, new_len) }
    }
}
