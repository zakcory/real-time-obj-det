use std::sync::Arc;
use tokio::sync::OnceCell;
use anyhow::{Context, Result};
use elasticsearch::{Elasticsearch, http::transport::Transport, BulkParts, http::request::JsonBody};
use serde_json::{json, Value};

// Custom modules
use crate::utils::config::{ElasticConfig, AppConfig};
use crate::utils::queue::FixedSizeQueue;
use crate::processing::ResultEmbedding;

// Constants
const MAX_QUEUE_SIZE: usize = 1000;
const FLUSH_INTERVAL: tokio::time::Duration = tokio::time::Duration::from_secs(2);

// Variables
pub static ELASTIC_CLIENT: OnceCell<Arc<Elastic>> = OnceCell::const_new();

/// Returns the elastic instance, if initiated
pub fn get_elastic() -> Result<&'static Arc<Elastic>> {
    Ok(
        ELASTIC_CLIENT
            .get()
            .context("Elastic client is not initiated!")?
    )
}

/// Initiates a single instance of elastic client
pub async fn init_elastic(app_config: &AppConfig) -> Result<()> {
    if let Some(_) = ELASTIC_CLIENT.get() {
        anyhow::bail!("Elastic client already initiated!")
    }

    // Create new instance
    let elastic_instance = Elastic::new(
        app_config.elastic_config().clone()
    )
        .context("Error creating new Elastic client")?;

    // Set global variable
    ELASTIC_CLIENT.set(Arc::new(elastic_instance))
        .map_err(|_| anyhow::anyhow!("Error setting Elastic client"))?;

    Ok(())
}

pub struct Elastic {
    queue: Arc<FixedSizeQueue<Value>>
}

impl Elastic {
    /// Creates a new Elastic client instance
    pub fn new(config: ElasticConfig) -> Result<Self> {
        let transport = Transport::single_node(&config.url)
            .context("Failed to create Elastic transport")?;
        
        let client = Elasticsearch::new(transport);
        
        // Create queue wrapped in Arc
        let queue = Arc::new(FixedSizeQueue::new(MAX_QUEUE_SIZE, None::<fn(Value)>));
        
        // Clone queue Arc for the background task
        let queue_clone = Arc::clone(&queue);
        let config_clone = config.clone();
        
        // Spawn background task for flushing
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(FLUSH_INTERVAL);
            
            loop {
                interval.tick().await;
                
                // Access receiver through the Arc
                let batch = queue_clone.receiver.drain().await;
                
                if !batch.is_empty() {
                    if let Err(e) = Self::send_bulk(&client, &config_clone, batch).await {
                        tracing::error!("Failed to send bulk request to Elastic: {:?}", e);
                    }
                }
            }
        });

        Ok(Self {
            queue,
        })
    }

    /// Adds embeddings to the queue
    pub async fn populate_embeddings(
        source_id: &str,
        timestamp: i64,
        embeddings: &[ResultEmbedding]
    ) -> Result<()> {
        let elastic = get_elastic()?;
        
        for embedding in embeddings {
            // Prepare document
            let doc = json!({
                "timestamp": timestamp,
                "channel_id": source_id,
                "embedding": embedding.data
            });
            
            // Send to queue
            elastic.queue.sender.send_async(doc).await;
        }

        Ok(())
    }

    /// Sends a bulk request to Elasticsearch
    async fn send_bulk(client: &Elasticsearch, config: &ElasticConfig, batch: Vec<Value>) -> Result<()> {
        let mut body: Vec<JsonBody<Value>> = Vec::with_capacity(batch.len() * 2);
        let total_embeddings = batch.len();

        for doc in batch {
            // Index action
            body.push(json!({"index": { "_index": config.index_name }}).into());
            // Document source
            body.push(doc.into());
        }

        let response = client
            .bulk(BulkParts::None)
            .body(body)
            .send()
            .await
            .context("Failed to send bulk request to Elastic")?;

        // Validate request response
        let response_body: Value = response.json().await?;
        if response_body["errors"].as_bool().unwrap_or(false) {
            // Log the actual error message from the first failed item
            tracing::error!("Bulk write contained errors: {:?}", response_body);
        } else {
            tracing::info!("Successfully sent bulk request to Elastic, Total {}", total_embeddings);
        }
        
        Ok(())
    }
}
