use axum::extract::Multipart;
use std::str::FromStr;
use utoipa::ToSchema;
use serde_json::json;
use axum::body::Bytes;

// Custom modules
use crate::processing::{self, RawFrame};
use crate::handlers::api::ApiResponse;
use crate::handlers::api::ApiResult;
use crate::utils::config::{SearchType, InferenceModelType};
use crate::utils::elastic::SearchMetadata;
use crate::handlers::TAG_MODELS;
use crate::utils;

#[derive(ToSchema)]
pub struct ImageSearchRequest {
    #[schema(value_type = String, format = Binary)]
    pub image: Bytes,
    pub search_type: SearchType,
    pub model_type: InferenceModelType,
    
    #[schema(nullable, example = json!(["1", "2"]))]
    pub channel_ids: Option<Vec<String>>,
    
    #[schema(nullable, example = json!(1767899017))]
    pub timestamp_start: Option<i64>,
    
    #[schema(nullable, example = json!(1767899017))]
    pub timestamp_end: Option<i64>,
}

impl ImageSearchRequest {
    async fn from_multipart(mut multipart: Multipart) -> Result<Self, String> {
        let mut image_data: Option<Bytes> = None;
        let mut search_type: Option<SearchType> = None;
        let mut model_type: Option<InferenceModelType> = None;
        let mut channel_ids: Option<Vec<String>> = None;
        let mut timestamp_start: Option<i64> = None;
        let mut timestamp_end: Option<i64> = None;

        while let Ok(Some(field)) = multipart.next_field().await {
            if let Some(name) = field.name() {
                if name == "image" {
                    match field.bytes().await {
                        Ok(bytes) => image_data = Some(bytes),
                        Err(_) => return Err(format!("Invalid image")),
                    }
                } else if name == "search_type" {
                    if let Ok(text) = field.text().await {
                        match SearchType::from_str(&text) {
                            Ok(st) => search_type = Some(st),
                            Err(_) => return Err(format!("Invalid search type: {}", text)),
                        }
                    }
                } else if name == "model_type" {
                    if let Ok(text) = field.text().await {
                        match InferenceModelType::from_str(&text) {
                            Ok(mt) => model_type = Some(mt),
                            Err(_) => return Err(format!("Invalid model type: {}", text)),
                        }
                    }
                } else if name == "channel_ids" {
                    if let Ok(text) = field.text().await {
                        if !text.is_empty() {
                            let ids = channel_ids.get_or_insert(Vec::new());
                            ids.push(text);
                        }
                    }
                } else if name == "timestamp_start" {
                    if let Ok(text) = field.text().await {
                        if !text.is_empty() {
                            if let Ok(ts) = text.parse::<i64>() {
                                timestamp_start = Some(ts);
                            }
                        }
                    }
                } else if name == "timestamp_end" {
                    if let Ok(text) = field.text().await {
                        if !text.is_empty() {
                            if let Ok(ts) = text.parse::<i64>() {
                                timestamp_end = Some(ts);
                            }
                        }
                    }
                }
            }
        }

        let image = image_data.ok_or("Missing image field")?;
        let search_type = search_type.ok_or("Missing search_type field")?;
        let model_type = model_type.ok_or("Missing model_type field")?;

        Ok(Self { 
            image, 
            search_type,
            model_type, 
            channel_ids, 
            timestamp_start, 
            timestamp_end 
        })
    }
}

/// Search for similar images
#[utoipa::path(
    post,
    path = "/search",
    tag = TAG_MODELS,
    request_body(content = ImageSearchRequest, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Search results"),
        (status = 400, description = "Invalid input")
    )
)]
pub async fn search_image(multipart: Multipart) -> ApiResult<serde_json::Value> {
    // Measure time of request
    let start = tokio::time::Instant::now();

    match ImageSearchRequest::from_multipart(multipart).await {
        Ok(request) => {
            match utils::parse_image(&request.image) {
                Ok((rgb_bytes, width, height)) => {
                    // Create raw frame instance
                    let raw_frame = RawFrame {
                        data: rgb_bytes,
                        height,
                        width,
                        added: start
                    };

                    // Construct SearchMetadata
                    let metadata = SearchMetadata {
                        channel_ids: request.channel_ids,
                        timestamp_start: request.timestamp_start,
                        timestamp_end: request.timestamp_end,
                    };

                    // Process frame and return search results
                    match processing::search::search_image(
                        raw_frame, 
                        request.search_type, 
                        request.model_type,
                        metadata
                    ).await {
                        Ok(candidates) => {
                            let results: Vec<serde_json::Value> = candidates.into_iter().map(|c| {
                                json!({
                                    "score": c["_score"],
                                    "metadata": c["_source"]
                                })
                            }).collect();

                            Ok(ApiResponse::success_with_message(
                                "Image processed successfully",
                                json!({
                                    "count": results.len(),
                                    "candidates": results
                                })
                            ))
                        },
                        Err(e) => {
                            tracing::error!(
                                error=?e,
                                "Could not process search"
                            );
                            
                            Ok(ApiResponse::bad_request("Error searching candidates"))
                        }
                    }
                },
                Err(_) => {
                    Ok(ApiResponse::bad_request("Error parsing image"))
                }
            }
        },
        Err(_) => {
            Ok(ApiResponse::bad_request("Could not process input"))
        }
    }
}
