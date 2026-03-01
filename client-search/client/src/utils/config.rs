//! Responsible for holding all application configuration under one place
//! for easy access and setting format for same variables

use std::path::{Path};
use std::collections::HashMap;
use anyhow::{self, Result, Context};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, fmt};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use serde_yaml;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use std::str::FromStr;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
pub enum SearchType {
    LIGHT,
    MEDIUM,
    HEAVY
}

impl FromStr for SearchType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "LIGHT" => Ok(SearchType::LIGHT),
            "MEDIUM" => Ok(SearchType::MEDIUM),
            "HEAVY" => Ok(SearchType::HEAVY),
            _ => Err(format!("Invalid search type: {}", s))
        }
    }
}

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
    pub instances: ModelInstancesConfig
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelInstancesConfig {
    pub default: u32,

    #[serde(default)]
    pub custom: HashMap<String, HashMap<InferenceModelType, u32>>
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
#[derive(PartialEq, Eq, Hash, Clone, Debug, Deserialize, ToSchema)]
#[allow(non_camel_case_types)]
pub enum InferenceModelType {
    SCENE,
    OBJECTS
}

impl InferenceModelType {
    pub fn to_string(&self) -> String {
        match self {
            InferenceModelType::SCENE => "SCENE".to_string(),
            InferenceModelType::OBJECTS => "OBJECTS".to_string()
        }
    }
}

impl FromStr for InferenceModelType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "SCENE" => Ok(InferenceModelType::SCENE),
            "OBJECTS" => Ok(InferenceModelType::OBJECTS),
            _ => Err(format!("Invalid inference model type: {}", s))
        }
    }
}

#[derive(PartialEq, Clone, Debug, Deserialize)]
pub struct SearchConfigOption {
    pub output_vectors: u32,
    pub num_candidates: u32,
    pub centriod_visit_percentage: u32,
    pub vector_oversample_multiplier: f32
}

/// Represents all the configuation variables used by the application
#[derive(Debug, Deserialize)]
pub struct AppConfig {
    local: bool,
    port: u16,
    elastic_config: ElasticConfig,
    triton_config: TritonConfig,
    inference_config: InferenceConfig,
    search_config: HashMap<SearchType, SearchConfigOption>
}

impl AppConfig {
    /// Creates a new instance of the configuration object
    pub fn new() -> Result<Self> {
        let config: AppConfig = AppConfig::load_config_file()
            .context("Error loading configuation file")?;

        // Initiate app logging
        AppConfig::init_logging(config.local);

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

    pub fn port(&self) -> u16 {
        self.port
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

    pub fn search_config(&self) -> &HashMap<SearchType, SearchConfigOption> {
        &self.search_config
    }
}