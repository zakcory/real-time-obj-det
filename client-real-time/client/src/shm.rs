use core::convert::From;
use std::collections::HashMap;
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
}

// `shared_memory::Shmem` owns an OS-backed shared-memory mapping and handle.
// Moving that ownership between async tasks/threads is fine as long as access
// remains synchronized by our surrounding logic.
unsafe impl Send for SharedMemoryRegion {}

impl SharedMemoryRegion {
    pub async fn register(client: &Client, prefix: &str, byte_size: usize) -> Result<Self> {
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
        };

        client
            .system_shared_memory_register(SystemSharedMemoryRegisterRequest {
                name: region_name.clone(),
                key: region.shmem.get_os_id().to_string(),
                offset: 0,
                byte_size: byte_size as u64,
            })
            .await
            .context("Error registering shared memory region with Triton")?;

        Ok(region)
    }

    pub async fn unregister(self, client: &Client) -> Result<()> {
        let region_name = self.region_name;
        client
            .system_shared_memory_unregister(SystemSharedMemoryUnregisterRequest {
                name: region_name.clone(),
            })
            .await
            .with_context(|| {
                format!(
                    "Error unregistering shared memory region {}",
                    region_name
                )
            })?;

        Ok(())
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
        let input: SharedMemoryRegion =
            SharedMemoryRegion::register(client, "input", input_byte_size).await?;

        let output: SharedMemoryRegion =
            match SharedMemoryRegion::register(client, "output", output_byte_size).await {
                Ok(region) => region,
                Err(err) => {
                    let _ = input.unregister(client).await?;
                    return Err(err);
                }
            };

        Ok(Self { input, output })
    }
}
