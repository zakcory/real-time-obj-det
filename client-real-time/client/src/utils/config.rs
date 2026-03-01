//! Responsible for holding all application configuration under one place
//! for easy access and setting format for same variables

use std::path::{Path};
use std::collections::HashMap;
use anyhow::{self, Result, Context};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, fmt};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use serde_yaml;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub precision: InferencePrecision,
    pub input_name: String,
    pub input_shape: Vec<i64>,
    pub output_name: String,
    pub output_shape: Vec<i64>,
    pub batch_max_size: u32,
    pub batch_max_queue_delay: u32,
    pub batch_preferred_sizes: Vec<u32>
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourcesConfig {
    #[serde(default)]
    pub sources: HashMap<String, SourceConfig>,
    pub ids: Vec<String>,
    pub default: SourceConfig,
    #[serde(default)]
    pub custom: HashMap<String, SourceConfigOptional>
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourceConfig {
    pub inf_frame: u32,
    pub conf_threshold: f32,
    pub nms_iou_threshold: f32
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourceConfigOptional {
    pub inf_frame: Option<u32>,
    pub conf_threshold: Option<f32>,
    pub nms_iou_threshold: Option<f32>
}

#[derive(Clone, Debug, Deserialize)]
pub struct TritonConfig {
    pub url: String
}

#[derive(Clone, Debug, Deserialize)]
pub struct ElasticConfig {
    pub url: String,
    pub index_name: String
}

#[derive(Clone, Debug, Deserialize)]
pub struct InferenceConfig {
    pub models: HashMap<InferenceModelType, ModelConfig>,
    pub task: InferenceTask
}

/// Represents the inference model precision type
#[derive(PartialEq, Eq, Clone, Copy, Debug, Deserialize)]
pub enum InferencePrecision {
    FP32,
    FP16
}

impl InferencePrecision {
    pub fn to_string(&self) -> String {
        match self {
            InferencePrecision::FP32 => "FP32".to_string(),
            InferencePrecision::FP16 => "FP16".to_string(),
        }
    }
}

/// Represents type of inference model
#[derive(PartialEq, Eq, Hash, Clone, Debug, Deserialize)]
#[allow(non_camel_case_types)]
pub enum InferenceModelType {
    YOLO,
    DINO,
    DINO_OBJECTS
}

impl InferenceModelType {
    pub fn to_string(&self) -> String {
        match self {
            InferenceModelType::YOLO => "YOLO".to_string(),
            InferenceModelType::DINO => "DINO".to_string(),
            InferenceModelType::DINO_OBJECTS => "DINO_OBJECTS".to_string()
        }
    }
}

/// Represents type of inference model
#[derive(Copy, Clone, Debug, Deserialize)]
pub enum InferenceTask {
    ObjectDetection,
    Embedding
}

/// Represents all the configuation variables used by the application
#[derive(Debug, Deserialize)]
pub struct AppConfig {
    local: bool,
    sources_config: SourcesConfig,
    elastic_config: ElasticConfig,
    triton_config: TritonConfig,
    inference_config: InferenceConfig
}

impl AppConfig {
    /// Creates a new instance of the configuration object
    pub fn new() -> Result<Self> {
        let mut config: AppConfig = AppConfig::load_config_file()
            .context("Error loading configuation file")?;

        // Initiate app logging
        AppConfig::init_logging(config.local);

        // Parse sources
        let mut sources: HashMap<String, SourceConfig> = HashMap::new();
        for source_id in config.sources_config().ids.iter() {
            // Get source preferred config
            let mut source_config = config.sources_config().default.clone();
            let custom_config = config.sources_config().custom.get(source_id);

            // Assign custom values - override defaults if exist
            source_config.inf_frame = custom_config
                .and_then(|o| o.inf_frame)
                .filter(|&x| x >= 1 && x <= 30)
                .unwrap_or(source_config.inf_frame);

            source_config.conf_threshold = custom_config
                .and_then(|o| o.conf_threshold)
                .filter(|&x| x >= 0.00 && x <= 1.00)
                .unwrap_or(source_config.conf_threshold);

            source_config.nms_iou_threshold = custom_config
                .and_then(|o| o.nms_iou_threshold)
                .filter(|&x| x >= 0.00 && x <= 1.00)
                .unwrap_or(source_config.nms_iou_threshold);

            sources.insert(
                source_id.clone(), 
                source_config
            );
        }
        config.sources_config.sources = sources;

        Ok(config)
    }

    /// Loads environment variables from a local .env file
    fn load_config_file() -> Result<AppConfig> {
        // Path relative to cwd
        let config_file = "secrets/config.yaml".to_string();
        let config_path = Path::new(&config_file);
        
        // Load configuration file
        let contents = std::fs::read_to_string(config_path)
            .context("Error locating configuration file")?;

        let config_file: AppConfig = serde_yaml::from_str(&contents)
            .context("Error parsing configuration file")?;

        Ok(config_file)
    }

    /// Initiates structured logging
    fn init_logging(local: bool) {
        let file_appender = RollingFileAppender::new(Rotation::NEVER, "logs", "app.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

        // Append logging to local file
        let file_layer = if local {
            Some(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_timer(fmt::time::UtcTime::rfc_3339())
                    .with_writer(non_blocking)
            )
        } else {
            None
        };

        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(
                // Console layer - pretty format
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_timer(fmt::time::UtcTime::rfc_3339())
                    .with_writer(std::io::stdout)
            )
            .with(file_layer)
            .init();

        std::mem::forget(_guard);
    }
}

impl AppConfig {
    pub fn is_local(&self) -> bool {
        self.local
    }

    pub fn sources_config(&self) -> &SourcesConfig {
        &self.sources_config
    }

    pub fn elastic_config(&self) -> &ElasticConfig {
        &self.elastic_config
    }

    pub fn triton_config(&self) -> &TritonConfig {
        &self.triton_config
    }

    pub fn inference_config(&self) -> &InferenceConfig {
        &self.inference_config
    }
}