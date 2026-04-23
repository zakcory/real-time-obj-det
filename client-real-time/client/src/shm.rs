use core::convert::From;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use shared_memory::{Shmem, ShmemConf};
use triton_client::Client;
use triton_client::inference::infer_parameter::ParameterChoice;
use triton_client::inference::{
    InferParameter,
    SystemSharedMemoryRegisterRequest,
    SystemSharedMemoryUnregisterRequest,
};

static SHM_REGION_COUNTER: AtomicU64 = AtomicU64::new(0);

const SHM_REGION_PARAM: &str = "shared_memory_region";
const SHM_OFFSET_PARAM: &str = "shared_memory_offset";
const SHM_BYTE_SIZE_PARAM: &str = "shared_memory_byte_size";

// Cap per (prefix, byte_size) bucket. Bounds pool growth if callers pass
// variable sizes; 16 leaves headroom for concurrent in-flight inferences
// sharing one bucket.
const MAX_REGIONS_PER_BUCKET: usize = 16;

#[derive(Hash, Eq, PartialEq)]
struct PoolKey {
    prefix: &'static str,
    byte_size: usize,
}

static POOL: OnceLock<Mutex<HashMap<PoolKey, Vec<SharedMemoryRegion>>>> = OnceLock::new();

fn pool() -> &'static Mutex<HashMap<PoolKey, Vec<SharedMemoryRegion>>> {
    // Pre-size to the typical number of buckets (one input, one output) so
    // the very first pool operations don't trigger a HashMap rehash.
    POOL.get_or_init(|| Mutex::new(HashMap::with_capacity(2)))
}

fn pool_take(prefix: &'static str, byte_size: usize) -> Option<SharedMemoryRegion> {
    let mut pool = pool().lock().ok()?;
    pool.get_mut(&PoolKey { prefix, byte_size })
        .and_then(|bucket| bucket.pop())
}

// Hands the region back to the pool. Returns `Some(region)` when the bucket
// is full (or the pool mutex is poisoned) so the caller falls through to a
// real Triton unregister; returns `None` once the region is parked.
fn pool_return(region: SharedMemoryRegion) -> Option<SharedMemoryRegion> {
    let key = PoolKey {
        prefix: region.prefix,
        byte_size: region.byte_size,
    };
    let mut pool = match pool().lock() {
        Ok(p) => p,
        Err(_) => return Some(region),
    };
    // On the first return per bucket, pre-size the Vec to the cap so the
    // subsequent pushes up to `MAX_REGIONS_PER_BUCKET` don't reallocate.
    let bucket = pool
        .entry(key)
        .or_insert_with(|| Vec::with_capacity(MAX_REGIONS_PER_BUCKET));
    if bucket.len() < MAX_REGIONS_PER_BUCKET {
        bucket.push(region);
        None
    } else {
        Some(region)
    }
}

fn next_region_name(prefix: &str) -> String {
    let counter = SHM_REGION_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}-{}-{}", prefix, std::process::id(), counter)
}

pub fn shm_params(region_name: &str, byte_size: usize) -> HashMap<String, InferParameter> {
    HashMap::from([
        (
            SHM_REGION_PARAM.to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::StringParam(region_name.to_string())),
            },
        ),
        (
            SHM_OFFSET_PARAM.to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::Int64Param(0)),
            },
        ),
        (
            SHM_BYTE_SIZE_PARAM.to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::Int64Param(byte_size as i64)),
            },
        ),
    ])
}

pub struct SharedMemoryRegion {
    pub region_name: String,
    pub byte_size: usize,
    pub shmem: Shmem,
    prefix: &'static str,
}

// A `SharedMemoryRegion` is owned by exactly one task at a time — either by
// the caller holding it after `register`, or by the pool. Handoff is by value
// and the pool's `Mutex` serializes access, so the inner `Shmem` is never
// aliased across threads. That's why sending across threads is sound even
// though `Shmem` is not `Send` by default.
unsafe impl Send for SharedMemoryRegion {}

impl SharedMemoryRegion {
    // Always allocates + Triton-registers a fresh region. Used by `register`
    // on pool miss and by `warm_pool` to eagerly pre-populate buckets.
    async fn create_and_register(
        client: &Client,
        prefix: &'static str,
        byte_size: usize,
    ) -> Result<Self> {
        let region_name = next_region_name(prefix);
        let shmem: Shmem = ShmemConf::new()
            .os_id(&region_name)
            .size(byte_size.max(1))
            .create()
            .context("Error creating shared memory region")?;

        let region = Self {
            region_name: region_name.clone(),
            byte_size,
            shmem,
            prefix,
        };

        // Pre-touch every page through the owning struct so neither the
        // first client-side write nor the first Triton-side write eats
        // demand-paging faults. SHM pages are shared via tmpfs, so faulting
        // them here populates the physical backing for both processes.
        // Accessing via `region.shmem` (which has `unsafe impl Send`) avoids
        // leaving a bare `Shmem` local visible to the async state machine,
        // which would otherwise make this future `!Send`.
        unsafe {
            std::ptr::write_bytes(region.shmem.as_ptr(), 0, region.shmem.len());
        }

        client
            .system_shared_memory_register(SystemSharedMemoryRegisterRequest {
                name: region_name,
                key: region.shmem.get_os_id().to_string(),
                offset: 0,
                byte_size: byte_size as u64,
            })
            .await
            .context("Error registering shared memory region with Triton")?;

        Ok(region)
    }

    // Always unregisters with Triton and releases the OS-backed segment.
    // Used by `unregister` on bucket-full eviction and by `drain_pool` on
    // graceful shutdown.
    async fn hard_unregister(self, client: &Client) -> Result<()> {
        let region_name = self.region_name;
        client
            .system_shared_memory_unregister(SystemSharedMemoryUnregisterRequest {
                name: region_name.clone(),
            })
            .await
            .with_context(|| format!("Error unregistering shared memory region {region_name}"))?;
        Ok(())
    }

    pub async fn register(
        client: &Client,
        prefix: &'static str,
        byte_size: usize,
    ) -> Result<Self> {
        // Fast path: reuse an already-registered region from the pool. No
        // syscalls, no Triton RPC — this is why SHM pays off.
        if let Some(region) = pool_take(prefix, byte_size) {
            return Ok(region);
        }
        Self::create_and_register(client, prefix, byte_size).await
    }

    pub async fn unregister(self, client: &Client) -> Result<()> {
        // Fast path: hand the region back to the pool. Keeps the Triton-side
        // registration and the mmap alive for the next inference.
        let Some(region) = pool_return(self) else {
            return Ok(());
        };
        // Eviction path: bucket is full, so actually unregister with Triton
        // and let `Shmem` drop (which shm_unlinks) to bound pool growth.
        region.hard_unregister(client).await
    }

    pub fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.byte_size {
            anyhow::bail!(
                "Shared memory write exceeds region size: got {} bytes, region has {} bytes",
                bytes.len(),
                self.byte_size
            )
        }

        unsafe {
            self.shmem.as_slice_mut()[..bytes.len()].copy_from_slice(bytes);
        }

        Ok(())
    }

    pub fn read_vec(&self, byte_size: usize) -> Result<Vec<u8>> {
        if byte_size > self.byte_size {
            anyhow::bail!(
                "Shared memory read exceeds region size: got {} bytes, region has {} bytes",
                byte_size,
                self.byte_size
            )
        }

        let bytes: &[u8] = unsafe { self.shmem.as_slice() };
        Ok(bytes[..byte_size].to_vec())
    }

    pub fn name(&self) -> &str {
        &self.region_name
    }
}

/// Pre-populate the pool with `count` regions of the given `(prefix, byte_size)`
/// so the first `count` inferences of that shape skip the `shm_open`/`mmap` +
/// Triton-register cost. `count` is clamped to `MAX_REGIONS_PER_BUCKET`.
///
/// Opt-in: call this from your Triton-client init path if cold-start latency
/// matters. Safe to call multiple times — anything beyond the bucket cap is
/// unregistered immediately instead of leaking.
pub async fn warm_pool(
    client: &Client,
    prefix: &'static str,
    byte_size: usize,
    count: usize,
) -> Result<()> {
    let capped = count.min(MAX_REGIONS_PER_BUCKET);
    let mut created: Vec<SharedMemoryRegion> = Vec::with_capacity(capped);
    for _ in 0..capped {
        created.push(SharedMemoryRegion::create_and_register(client, prefix, byte_size).await?);
    }
    for region in created {
        if let Some(overflow) = pool_return(region) {
            overflow.hard_unregister(client).await?;
        }
    }
    Ok(())
}

/// Drain every region currently parked in the pool, Triton-unregistering each
/// and releasing its OS-backed segment. Call on graceful shutdown — Rust does
/// not run `Drop` on statics, so without this the Triton registrations and
/// `/dev/shm/*` files persist until the kernel reaps the process.
pub async fn drain_pool(client: &Client) {
    let Some(pool_mutex) = POOL.get() else {
        return;
    };
    let drained: Vec<SharedMemoryRegion> = match pool_mutex.lock() {
        Ok(mut pool) => pool.drain().flat_map(|(_, v)| v).collect(),
        Err(_) => return,
    };
    // Unregister concurrently so shutdown doesn't scale linearly with the
    // number of pooled regions (each unregister is a Triton RPC).
    let results = futures::future::join_all(
        drained.into_iter().map(|region| region.hard_unregister(client)),
    )
    .await;
    for result in results {
        if let Err(err) = result {
            tracing::warn!("Failed to unregister pooled region during drain: {err:?}");
        }
    }
}

pub struct ShmRequest {
    pub input: SharedMemoryRegion,
    pub output: SharedMemoryRegion,
}

impl ShmRequest {
    pub async fn new(
        client: &Client,
        input_byte_size: usize,
        output_byte_size: usize,
    ) -> Result<Self> {
        // Run both registrations concurrently. In steady state both futures
        // complete synchronously from the pool and `join!` is effectively
        // free; on cold miss the two Triton RPCs overlap and the cold-start
        // cost drops from 2 RTTs to 1.
        let (input_res, output_res) = tokio::join!(
            SharedMemoryRegion::register(client, "input", input_byte_size),
            SharedMemoryRegion::register(client, "output", output_byte_size),
        );

        match (input_res, output_res) {
            (Ok(input), Ok(output)) => Ok(Self { input, output }),
            (Ok(input), Err(err)) => {
                if let Err(cleanup_err) = input.unregister(client).await {
                    tracing::warn!(
                        "Failed to return input region to pool after output register failure: {cleanup_err:?}"
                    );
                }
                Err(err)
            }
            (Err(err), Ok(output)) => {
                if let Err(cleanup_err) = output.unregister(client).await {
                    tracing::warn!(
                        "Failed to return output region to pool after input register failure: {cleanup_err:?}"
                    );
                }
                Err(err)
            }
            (Err(err), Err(_)) => Err(err),
        }
    }
}
