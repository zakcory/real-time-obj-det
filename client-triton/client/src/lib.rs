use anyhow::{Result, Context};
use tokio::runtime::Handle;
use tokio::sync::OnceCell;

// Custom modules
pub mod utils;
pub mod inference;
pub mod processing;
pub mod client_video;
pub mod source;
pub mod statistics;

pub static TOKIO_RUNTIME: OnceCell<Handle> = OnceCell::const_new();

/// Getting the global tokio runtime context for functions called outside
/// of our application. Allows us to send those functions into our threadpools
/// and execute them on our threads
pub fn get_tokio_runtime() -> Result<&'static Handle> {
    Ok(
        TOKIO_RUNTIME
            .get()
            .context("Tokio runtime is not set")?
    )
}

pub async fn init_tokio_runtime(handle: Handle) -> Result<()> {
    if let Ok(_) = get_tokio_runtime() {
        anyhow::bail!("Tokio runtime is already set")
    }

    // Set global variable
    TOKIO_RUNTIME.set(handle)
        .map_err(|_| anyhow::anyhow!("Error setting tokio runtime"))?;

    Ok(())
}