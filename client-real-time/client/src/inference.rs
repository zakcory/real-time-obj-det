///! Responsible for performing inference with Nvidia Triton Server
///! 
///! Performs operations using gRPC protocol for minimal latency between
///! our application and Triton Server.
///! Allows us to dynamically load models(multiple instances) depending on amount of video sources we have

use triton_client::Client;
use triton_client::inference::{ModelInferRequest, RepositoryModelLoadRequest, ModelRepositoryParameter, RepositoryModelUnloadRequest};
use triton_client::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use triton_client::inference::model_repository_parameter::{ParameterChoice};
use std::collections::HashMap;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::OnceCell;
use anyhow::{self, Result, Context};

// Custom modules
use crate::utils::config::{AppConfig, ModelConfig, TritonConfig, InferenceModelType, InferencePrecision};

// Variables
pub static INFERENCE_MODELS: OnceCell<HashMap<InferenceModelType, Arc<InferenceModel>>> = OnceCell::const_new();

/// Returns the inference model instance, if initiated
pub fn get_inference_model(model_type: InferenceModelType) -> Result<&'static Arc<InferenceModel>> {
    Ok(
        INFERENCE_MODELS
            .get()
            .context("Infernece models are not initiated!")?
            .get(&model_type)
            .context("Infernece model is not initiated!")?
    )
}

/// Initiates a single instance of a model for inference
pub async fn init_inference_models(app_config: &AppConfig) -> Result<()> {
    if let Some(_) = INFERENCE_MODELS.get() {
        anyhow::bail!("Models are already initiated!")
    }

    // Create model instances
    let mut models: HashMap<InferenceModelType, Arc<InferenceModel>> = HashMap::new();
    for (model_type, model_config) in app_config.inference_config().models.iter() {
        // Create single instance
        let client_instance = InferenceModel::new(
            app_config.triton_config().clone(),
            model_config.clone(),
        )
            .await
            .context("Error creating model client")?;

        models.insert(
            model_type.clone(),
            Arc::new(client_instance)
        );
    }

    // Set global variable
    INFERENCE_MODELS.set(models)
        .map_err(|_| anyhow::anyhow!("Error setting model instances"))?;

    Ok(())
}

pub async fn start_models_instances(app_config: &AppConfig) -> Result<()> {
    // Calculate total "load units" - how much processing capacity we need
    // Each source contributes fractional load based on its frame rate
    let instances: u32 = app_config
        .sources_config()
        .sources
        .len() as u32;

    // Load same amount of instances for each model type
    for model_type in app_config.inference_config().models.keys() {
        let client_instance = get_inference_model(model_type.clone())?;

        // Clear previous model instances
        if let Ok(_) = client_instance.unload_model().await {
            tracing::warn!("Unloaded previous model instances for type {}", model_type.to_string());
        }

        // Initiate model instances
        client_instance.load_model(instances).await
            .context("Error loading model instances")?;

        tracing::info!("Initiated {} model instances for type {}", instances, model_type.to_string());
    }

    Ok(())
}

/// Represents an instance of an inference model
pub struct InferenceModel {
    client: Arc<Client>,
    triton_config: TritonConfig,
    model_config: ModelConfig,
    base_request: ModelInferRequest
}

impl InferenceModel {
    /// Create new instance of inference model
    /// 
    /// Creates a new Triton Server client for inference
    /// Initiate all values for fast inference, including a pre-made request body for inference
    /// Reports statistics about GPU utilization
    pub async fn new(
        triton_config: TritonConfig,
        model_config: ModelConfig
    ) -> Result<Self> {
        //Create client instance
        let client = Client::new(&triton_config.url, None)
            .await
            .context("Error creating triton client instance")?;

        // Check if server is ready
        let server_ready = client.server_ready()
            .await
            .context("Error getting model ready status")?;

        if !server_ready.ready {
            anyhow::bail!("Triton server is not ready");
        }

        // Create base inference request
        let mut batch_input_shape = Vec::with_capacity(&model_config.input_shape.len() + 1);
        batch_input_shape.extend(&model_config.input_shape);

        let base_request = ModelInferRequest {
            model_name: model_config.name.to_string(),
            model_version: "1".to_string(),
            id: String::new(),
            parameters: HashMap::new(),
            inputs: vec![
                InferInputTensor {
                    name: model_config.input_name.to_string(),
                    datatype: model_config.precision.to_string(),
                    shape: batch_input_shape,
                    parameters: HashMap::new(),
                    contents: None
                }
            ],
            outputs: vec![
                InferRequestedOutputTensor {
                    name: model_config.output_name.to_string(),
                    parameters: HashMap::new(),
                }
            ],
            raw_input_contents: Vec::new()
        };

        Ok(Self { 
            client: Arc::new(client),
            triton_config,
            model_config,
            base_request
        })
    }

    /// Unloads running instances of a given model
    pub async fn unload_model(&self) -> Result<()> {
        // Unload previous instances of model we're about to load
        self.client.repository_model_unload(RepositoryModelUnloadRequest { 
            repository_name: "".to_string(), 
            model_name: self.model_config().name.to_string(), 
            parameters: HashMap::new()
        })
            .await
            .context("Error unloading previous triton model instances")?;

        Ok(())
    }
    
    /// Loads given amount of instances of a given model
    pub async fn load_model(&self, instances: u32) -> Result<()> {
        let model_config = json!({
            "name": &self.model_config().name,
            "platform": "tensorrt_plan",
            "max_batch_size": &self.model_config().batch_max_size,
            "input": [
                {
                    "name": &self.model_config().input_name,
                    "data_type": format!("TYPE_{}", &self.model_config().precision.to_string()),
                    "dims": &self.model_config().input_shape
                }
            ],
            "output": [
                {
                    "name": &self.model_config().output_name,
                    "data_type": format!("TYPE_{}", &self.model_config().precision.to_string()),
                    "dims": &self.model_config().output_shape
                }
            ],
            "instance_group": [
                {
                    "kind": "KIND_GPU",
                    "count": instances,
                    "gpus": [0]
                }
            ],
            "dynamic_batching": {
                "max_queue_delay_microseconds": self.model_config().batch_max_queue_delay,
                "preferred_batch_size": &self.model_config().batch_preferred_sizes,
                "preserve_ordering": false
            },
            "optimization": {
                "execution_accelerators": {
                "gpu_execution_accelerator": [
                    {
                        "name": "tensorrt",
                        "parameters": {
                            "key": "precision_mode",
                            "value": &self.model_config().precision.to_string()
                        }
                    }
                ]
                },
                "input_pinned_memory": {
                    "enable": true
                },
                "output_pinned_memory": {
                    "enable": true
                },
                "gather_kernel_buffer_threshold": 0
            },
            "model_transaction_policy": {
                "decoupled": false
            },
            "model_warmup": [
                {
                    "name": "warmup_random",
                    "batch_size": self.model_config().batch_max_size,
                    "inputs":  {
                        &self.model_config().input_name: {
                            "dims": &self.model_config().input_shape,
                            "data_type": format!("TYPE_{}", &self.model_config().precision.to_string()),
                            "random_data": true
                        }
                    }
                }
            ]
        });

        // Define model config
        let mut parameters = HashMap::new();
        parameters.insert("config".to_string(), ModelRepositoryParameter{ 
            parameter_choice: Some(ParameterChoice::StringParam(model_config.to_string()))
        });

        // Load selected model
        self.client.repository_model_load(RepositoryModelLoadRequest { 
            repository_name: "".to_string(), 
            model_name: self.model_config().name.to_string(), 
            parameters
        })
            .await
            .context("Error loading triton model instances")?;

        Ok(())
    }

    /// Performs inference on many raw inputs, returning raw model results
    /// Automatically batches requests up to max_batch_size and processes batches concurrently
    pub async fn infer(&self, raw_inputs: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>> {
        let max_batch_size = self.model_config.batch_max_size as usize;
        let num_inputs = raw_inputs.len();        
        // Calculate output size per sample once
        let output_size_per_sample: usize = self.model_config.output_shape
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>() * match self.model_config.precision {
                InferencePrecision::FP16 => 2,
                InferencePrecision::FP32 => 4,
            };
        
        // Pre-allocate result slots - direct placement, no sorting
        let mut all_results: Vec<Vec<u8>> = Vec::with_capacity(num_inputs);
        all_results.resize_with(num_inputs, Vec::new);
        
        // Fast path: if inputs fit in one batch, execute directly without spawning tasks
        if num_inputs <= max_batch_size {
            let total_bytes: usize = raw_inputs.iter().map(|v| v.len()).sum();
            let mut concatenated = Vec::with_capacity(total_bytes);
            for input in &raw_inputs {
                concatenated.extend_from_slice(input);
            }
            
            let mut inference_request = self.base_request.clone();
            inference_request.inputs[0].shape.insert(0, num_inputs as i64);
            inference_request.raw_input_contents = vec![concatenated];
            
            // Network I/O - direct await
            let inference_result = self.client.model_infer(inference_request)
                .await
                .context("Error sending triton inference request")?;
            
            let output_blob = inference_result.raw_output_contents.into_iter().next()
                .context("No output from inference")?;
            
            // Process results directly
            // We do this inline because for a single batch the overhead of spawning a blocking task
            // might outweigh the benefit, and we want to minimize latency.
            let ptr = output_blob.as_ptr();
            unsafe {
                for i in 0..num_inputs {
                    let offset = i * output_size_per_sample;
                    let slice = std::slice::from_raw_parts(ptr.add(offset), output_size_per_sample);
                    all_results[i] = slice.to_vec();
                }
            }
        } else {     
            // Process all batches concurrently (1 batch if num_inputs <= max_batch_size)
            let tasks: Vec<_> = raw_inputs
                .chunks(max_batch_size)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let batch_size = chunk.len();
                    let start_idx = chunk_idx * max_batch_size;
                    
                    // Concatenate batch for Triton
                    let total_bytes: usize = chunk.iter().map(|v| v.len()).sum();
                    let mut concatenated = Vec::with_capacity(total_bytes);
                    for input in chunk {
                        concatenated.extend_from_slice(input);
                    }
                    
                    let mut inference_request = self.base_request.clone();
                    inference_request.inputs[0].shape.insert(0, batch_size as i64);
                    inference_request.raw_input_contents = vec![concatenated];
                    
                    let client = Arc::clone(&self.client);
                    let output_size = output_size_per_sample;
                    
                    tokio::spawn(async move {
                        // Network I/O - async
                        let inference_result = client.model_infer(inference_request)
                            .await
                            .context("Error sending triton inference request")?;
                        
                        // CPU work - blocking thread pool
                        let output_blob = inference_result.raw_output_contents.into_iter().next()
                            .context("No output from inference")?;
                        
                        let batch_results = tokio::task::spawn_blocking(move || {
                            // Unsafe pointer slicing for blazing speed
                            let ptr = output_blob.as_ptr();
                            let mut results = Vec::with_capacity(batch_size);
                            
                            unsafe {
                                for i in 0..batch_size {
                                    let offset = i * output_size;
                                    let slice = std::slice::from_raw_parts(ptr.add(offset), output_size);
                                    results.push(slice.to_vec());
                                }
                            }
                            
                            results
                        })
                        .await
                        .context("Failed to split batch results")?;
                        
                        Ok::<(usize, Vec<Vec<u8>>), anyhow::Error>((start_idx, batch_results))
                    })
                })
                .collect();
            
            // Await all batches and place directly
            let results = futures::future::try_join_all(tasks)
                .await
                .context("Error performing inference on all inputs")?;
            
            for result in results {
                let (start_idx, batch) = result?;
                for (i, output) in batch.into_iter().enumerate() {
                    all_results[start_idx + i] = output;
                }
            }
        }

        Ok(all_results)
    }
}

impl InferenceModel {
    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn triton_config(&self) -> &TritonConfig {
        &self.triton_config
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    pub fn base_request(&self) -> &ModelInferRequest {
        &self.base_request
    }
}