use anyhow::{Result, Context};

// Custom modules
use client::{source, inference, statistics};
use client::utils::{
    config::AppConfig,
    elastic
};
use client::client_video::{self, ClientVideo};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // Iniaitlize config
    let app_config = AppConfig::new()
        .context("Error loading config")?;

    client::init_tokio_runtime(tokio::runtime::Handle::current())
        .await
        .context("Error initializing tokio runtime")?;

    elastic::init_elastic(&app_config)
        .await
        .context("Error initiating elastic")?;

    // Initiate inference client
    inference::init_inference_models(&app_config)
        .await
        .context("Error initiating inference model")?;

    inference::start_models_instances(&app_config)
        .await
        .context("Error initiating inference model instances")?;

    // Initiate sources processors
    source::init_source_processors(&app_config)
        .await
        .context("Error initiating source processors")?;

    // Initiate statistics
    statistics::init_statistics()
        .context("Error initiating statistics")?;

    // Start receiving frames from sources
    client_video::init_client_video()
        .context("Error initiating client video")?;

    ClientVideo::init_sources(&app_config)
        .await
        .context("Error setting Client Video callbacks")?;

    Ok(())
}
