use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::OnceCell;
use anyhow::{Context, Result};
use elasticsearch::{Elasticsearch, http::transport::Transport};
use serde_json::json;

// Custom modules
use crate::utils::config::{ElasticConfig, AppConfig, SearchType, SearchConfigOption};
use crate::processing::ResultEmbedding;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

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
        app_config.elastic_config().clone(),
        app_config.search_config().clone()
    )
        .context("Error creating new Elastic client")?;

    // Set global variable
    ELASTIC_CLIENT.set(Arc::new(elastic_instance))
        .map_err(|_| anyhow::anyhow!("Error setting Elastic client"))?;

    Ok(())
}

pub struct Elastic {
    client: Elasticsearch,
    elastic_config: ElasticConfig,
    search_config: HashMap<SearchType, SearchConfigOption>
}

#[derive(Debug, Deserialize, Serialize, Clone, ToSchema)]
pub struct SearchMetadata {
    pub channel_ids: Option<Vec<String>>,
    pub timestamp_start: Option<i64>,
    pub timestamp_end: Option<i64>
}

impl Elastic {
    /// Creates a new Elastic client instance
    pub fn new(
        elastic_config: ElasticConfig, 
        search_config: HashMap<SearchType, SearchConfigOption>
    ) -> Result<Self> {
        let transport = Transport::single_node(&elastic_config.url)
            .context("Failed to create Elastic transport")?;
        
        let client = Elasticsearch::new(transport);

        Ok(Self {
            client,
            elastic_config,
            search_config
        })
    }

    /// Performs a KNN search on the index with the given embedding and configuration
    pub async fn search_disk_bbq(
        embedding: ResultEmbedding,
        search_type: SearchType,
        metadata: SearchMetadata
    ) -> Result<Vec<serde_json::Value>> {
        let elastic = get_elastic()?;

        // Determine search configuration
        let search_config = elastic.search_config.get(&search_type)
            .context("Search configuration not found")?;

        let mut must_clauses = Vec::new();

        // Filter by channel IDs
        if let Some(ids) = metadata.channel_ids {
            if !ids.is_empty() {
                must_clauses.push(json!({
                    "terms": {
                        "channel_id": ids
                    }
                }));
            }
        }

        // Filter by timestamps (start and end)
        if metadata.timestamp_start.is_some() || metadata.timestamp_end.is_some() {
            let mut range_query = json!({});
            
            if let Some(start) = metadata.timestamp_start {
                range_query["gte"] = json!(start);
            }
            
            if let Some(end) = metadata.timestamp_end {
                range_query["lte"] = json!(end);
            }

            must_clauses.push(json!({
                "range": {
                    "timestamp": range_query
                }
            }));
        }

        let filter = if !must_clauses.is_empty() {
            Some(json!({
                "bool": {
                    "must": must_clauses
                }
            }))
        } else {
            None
        };

        let mut knn_query = json!({
            "field": "embedding",
            "query_vector": embedding.data,
            "k": search_config.output_vectors,
            "num_candidates": search_config.num_candidates,
            "visit_percentage": search_config.centriod_visit_percentage,
            "rescore_vector": {
                "oversample": search_config.vector_oversample_multiplier
            }
        });

        if let Some(f) = filter {
            knn_query["filter"] = f;
        }

        let body = json!({
            "timeout": "25s",
            "size": search_config.output_vectors,
            "knn": knn_query,
            "_source": {
                "excludes": ["embedding"]
            }
        });

        let response = elastic.client
            .search(elasticsearch::SearchParts::Index(&[&elastic.elastic_config.index_name]))
            .body(body)
            .send()
            .await
            .context("Failed to execute search request")?;

        let response_body = response.json::<serde_json::Value>().await
            .context("Failed to parse search response")?;

        // Return search results
        let hits = response_body["hits"]["hits"]
            .as_array()
            .map(|h| h.to_vec())
            .unwrap_or_default();

        Ok(hits)
    }
}
