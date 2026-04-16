//! Responsible for holding all application configuration under one place
//! for easy access and setting format for same variables

use anyhow::{self, Context, Result};
use serde::Deserialize;
use serde_yaml;
use std::collections::HashMap;
use std::path::Path;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

const PACKED_NMS_OUTPUT_NAME: &str = "det_packed";

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub precision: InferencePrecision,
    pub input_name: String,
    pub input_shape: Vec<i64>,
    #[serde(default = "default_nms_in_triton")]
    pub nms_in_triton: bool,
    #[serde(default = "default_use_shm")]
    pub use_shm: bool,
    #[serde(default)]
    pub output_name: Option<String>,
    #[serde(default)]
    pub output_shape: Option<Vec<i64>>,
    #[serde(default)]
    pub outputs: Vec<ModelOutputConfig>,
    pub batch_max_size: u32,
    pub batch_max_queue_delay: u32,
    pub batch_preferred_sizes: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelOutputConfig {
    pub name: String,
    pub data_type: InferencePrecision,
    pub shape: Vec<i64>,
}

fn default_nms_in_triton() -> bool {
    true
}

fn default_use_shm() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourcesConfig {
    #[serde(default)]
    pub sources: HashMap<String, SourceConfig>,
    pub ids: Vec<String>,
    pub default: SourceConfig,
    #[serde(default)]
    pub custom: HashMap<String, SourceConfigOptional>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourceConfig {
    pub inf_frame: u32,
    pub conf_threshold: f32,
    pub nms_iou_threshold: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourceConfigOptional {
    pub inf_frame: Option<u32>,
    pub conf_threshold: Option<f32>,
    pub nms_iou_threshold: Option<f32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TritonConfig {
    pub url: String,
    #[serde(default = "default_use_shm")]
    pub use_shm: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ElasticConfig {
    pub url: String,
    pub index_name: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InferenceConfig {
    pub models: HashMap<InferenceModelType, ModelConfig>,
    pub task: InferenceTask,
    pub instances: u32,
}

/// Represents the inference model precision type
#[derive(PartialEq, Eq, Clone, Copy, Debug, Deserialize)]
pub enum InferencePrecision {
    #[serde(alias = "TYPE_FP32")]
    FP32,
    #[serde(alias = "TYPE_FP16")]
    FP16,
}

impl InferencePrecision {
    pub fn to_string(&self) -> String {
        match self {
            InferencePrecision::FP32 => "FP32".to_string(),
            InferencePrecision::FP16 => "FP16".to_string(),
        }
    }

    pub fn to_triton_data_type(&self) -> String {
        format!("TYPE_{}", self.to_string())
    }

    pub fn byte_size(&self) -> usize {
        match self {
            InferencePrecision::FP32 => 4,
            InferencePrecision::FP16 => 2,
        }
    }
}

/// Represents type of inference model
#[derive(PartialEq, Eq, Hash, Clone, Debug, Deserialize)]
#[allow(non_camel_case_types)]
pub enum InferenceModelType {
    YOLO,
    DINO,
    DINO_OBJECTS,
}

impl InferenceModelType {
    pub fn to_string(&self) -> String {
        match self {
            InferenceModelType::YOLO => "YOLO".to_string(),
            InferenceModelType::DINO => "DINO".to_string(),
            InferenceModelType::DINO_OBJECTS => "DINO_OBJECTS".to_string(),
        }
    }
}

/// Represents type of inference model
#[derive(Copy, Clone, Debug, Deserialize)]
pub enum InferenceTask {
    ObjectDetection,
    Embedding,
}

/// Represents all the configuation variables used by the application
#[derive(Debug, Deserialize)]
pub struct AppConfig {
    local: bool,
    sources_config: SourcesConfig,
    elastic_config: ElasticConfig,
    triton_config: TritonConfig,
    inference_config: InferenceConfig,
}

impl AppConfig {
    /// Creates a new instance of the configuration object
    pub fn new() -> Result<Self> {
        let mut config: AppConfig =
            AppConfig::load_config_file().context("Error loading configuation file")?;

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

            sources.insert(source_id.clone(), source_config);
        }
        config.sources_config.sources = sources;

        for (model_type, model_config) in config.inference_config().models.iter() {
            model_config.resolved_outputs().with_context(|| {
                format!(
                    "Invalid output configuration for inference model {}",
                    model_type.to_string()
                )
            })?;
        }

        Ok(config)
    }

    /// Loads environment variables from a local .env file
    fn load_config_file() -> Result<AppConfig> {
        // Path relative to cwd
        let config_file = "secrets/config.yaml".to_string();
        let config_path = Path::new(&config_file);

        // Load configuration file
        let contents =
            std::fs::read_to_string(config_path).context("Error locating configuration file")?;

        let config_file: AppConfig =
            serde_yaml::from_str(&contents).context("Error parsing configuration file")?;

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
                    .with_writer(non_blocking),
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
                    .with_writer(std::io::stdout),
            )
            .with(file_layer)
            .init();

        std::mem::forget(_guard);
    }
}

impl ModelConfig {
    pub fn nms_in_triton(&self) -> bool {
        self.nms_in_triton
    }

    pub fn resolved_outputs(&self) -> Result<Vec<ModelOutputConfig>> {
        let outputs = if !self.outputs.is_empty() {
            self.outputs.clone()
        } else {
            let output_name = self
                .output_name
                .clone()
                .context("Missing model output_name configuration")?;
            let output_shape = self
                .output_shape
                .clone()
                .context("Missing model output_shape configuration")?;

            let data_type = if self.nms_in_triton {
                InferencePrecision::FP32
            } else {
                self.precision
            };

            vec![ModelOutputConfig {
                name: output_name,
                data_type,
                shape: output_shape,
            }]
        };

        if self.nms_in_triton {
            if outputs.len() != 1 {
                anyhow::bail!(
                    "Packed NMS model contract expects exactly one output, got {}",
                    outputs.len()
                );
            }

            let packed_output = &outputs[0];
            if packed_output.name != PACKED_NMS_OUTPUT_NAME {
                anyhow::bail!(
                    "Packed NMS model output must be named {}, got {}",
                    PACKED_NMS_OUTPUT_NAME,
                    packed_output.name
                );
            }

            if packed_output.data_type != InferencePrecision::FP32 {
                anyhow::bail!(
                    "Packed NMS model output must use FP32, got {}",
                    packed_output.data_type.to_string()
                );
            }

            if packed_output.shape.len() != 1 {
                anyhow::bail!(
                    "Packed NMS model output must have 1D non-batch shape [1 + 6*K], got {:?}",
                    packed_output.shape
                );
            }

            let packed_len = packed_output.shape[0];
            if packed_len <= 1 || ((packed_len - 1) % 6) != 0 {
                anyhow::bail!(
                    "Packed NMS model output shape must match [1 + 6*K], got {:?}",
                    packed_output.shape
                );
            }
        }

        Ok(outputs)
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
