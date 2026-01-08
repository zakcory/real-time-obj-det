use anyhow::Result;

// Custom modules
use crate::utils::config::{SearchType, InferenceModelType};
use crate::utils::elastic::SearchMetadata;
use crate::processing::RawFrame;
use std::sync::Arc;

// Custom modules
use crate::inference;
use crate::processing;
use crate::utils::elastic::Elastic;
use crate::statistics;

pub async fn search_image(
    raw_frame: RawFrame, 
    search_type: SearchType, 
    model_type: InferenceModelType,
    metadata: SearchMetadata
) -> Result<Vec<serde_json::Value>> {
    let frame_queue_time = raw_frame.added.elapsed();
    let raw_frame = Arc::new(raw_frame);

    // Perform inference on the frame
    let inference_model = inference::get_inference_model(model_type)?;
    let frame = Arc::clone(&raw_frame);
    let (mut inference_stats, embedding) = processing::dino::process_frame(
        &inference_model, 
        frame
    ).await?;

    // Search for similar images
    let measure_start = tokio::time::Instant::now();
    let search_results = Elastic::search_disk_bbq(
        embedding,
        search_type, 
        metadata
    ).await?;

    let search_time = measure_start.elapsed();
    inference_stats.search += search_time.as_micros() as u64;

    // Add final statistics
    inference_stats.queue = frame_queue_time.as_micros() as u64;
    inference_stats.processing += frame_queue_time.as_micros() as u64;

    // Add statistics to global counters
    statistics::get_statistics()?
        .processing_stats()
        .accumulate(&inference_stats);
    
    Ok(search_results)
}