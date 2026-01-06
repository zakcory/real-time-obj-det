/// Module for DINOv3 model pre/post processing

use anyhow::{Result, Context};
use std::sync::Arc;
use std::time::Instant;

// Custom modules
use crate::inference::InferenceModel;
use crate::statistics::FrameProcessStats;
use crate::processing::{self, RawFrame, ResultEmbedding, ResultBBOX};
use crate::utils::config::InferencePrecision;

/// Performs pre-processing on raw RGB frame for DINOv3 model
/// 
/// This function performs pre-processing steps including resizing, center cropping,
/// and normalization(pixel & ImageNet) to prepare the frame for inference with DINOv3 models.
pub fn preprocess_frame(
    frame: &RawFrame,
    precision: InferencePrecision,
) -> Result<Vec<u8>> {
    // Validate input
    let frame_target_size = (frame.height * frame.width * 3) as usize;
    if frame.data.len() != frame_target_size {
        anyhow::bail!(
            "Got unexpected size of frame input. Got {}, expected {}",
            frame.data.len(),
            frame_target_size
        );
    }

    // Preprocess with letterbox resize + ImageNet normalization
    const TARGET_SIZE: u32 = 224;
    processing::resize_letterbox_and_normalize_imagenet(
        &frame.data,
        frame.height,
        frame.width,
        TARGET_SIZE,
        TARGET_SIZE,
        precision
    )
}

/// Performs post-processing on multiple raw inference results from DINOv3 models
/// 
/// Takes a Vec of raw Vec<u8> outputs from batch model inference and converts them to 
/// a Vec of ResultEmbedding containing the feature vectors.
pub fn postprocess(
    raw_results: Vec<Vec<u8>>,
    precision: InferencePrecision,
) -> Result<Vec<ResultEmbedding>> {
    let mut embeddings = Vec::with_capacity(raw_results.len());
    
    for raw_result in raw_results {
        let num_elements = match precision {
            InferencePrecision::FP16 => raw_result.len() / 2,
            InferencePrecision::FP32 => raw_result.len() / 4,
        };
        
        let embedding = match precision {
            InferencePrecision::FP16 => {
                let raw_ptr = raw_result.as_ptr() as *const u16;
                let mut data = Vec::with_capacity(num_elements);
                unsafe {
                    for i in 0..num_elements {
                        data.push(processing::get_f16_to_f32_lut(*raw_ptr.add(i)));
                    }
                }
                ResultEmbedding { data }
            }
            InferencePrecision::FP32 => {
                let raw_ptr = raw_result.as_ptr() as *const f32;
                let data = unsafe {
                    Vec::from_raw_parts(
                        raw_ptr as *mut f32,
                        num_elements,
                        num_elements
                    )
                };
                std::mem::forget(raw_result);
                ResultEmbedding { data }
            }
        };
        
        embeddings.push(embedding);
    }
    
    Ok(embeddings)
}

/// Preprocesses bounding boxes from a frame for DINOv3 inference
/// 
/// Crops each bbox region from the frame, applies letterbox resizing with padding,
/// and performs ImageNet normalization to prepare for DINOv3 model input.
pub fn preprocess_bbox(
    frame: &RawFrame,
    bbox: &ResultBBOX,
    precision: InferencePrecision,
) -> Result<Vec<u8>> {
    // Extract bbox coordinates [x1, y1, x2, y2]
    let mut b_x1 = bbox.bbox[0];
    let mut b_y1 = bbox.bbox[1];
    let mut b_x2 = bbox.bbox[2];
    let mut b_y2 = bbox.bbox[3];

    // Stretch bbox by 1.5x
    let width = b_x2 - b_x1;
    let height = b_y2 - b_y1;
    let cx = b_x1 + width / 2.0;
    let cy = b_y1 + height / 2.0;
    
    let new_width = width * 1.5;
    let new_height = height * 1.5;
    
    b_x1 = cx - new_width / 2.0;
    b_y1 = cy - new_height / 2.0;
    b_x2 = cx + new_width / 2.0;
    b_y2 = cy + new_height / 2.0;

    let x1 = b_x1.max(0.0) as u32;
    let y1 = b_y1.max(0.0) as u32;
    let x2 = b_x2.min(frame.width as f32) as u32;
    let y2 = b_y2.min(frame.height as f32) as u32;
    
    // Calculate bbox dimensions
    let bbox_width = x2.saturating_sub(x1);
    let bbox_height = y2.saturating_sub(y1);
    
    // Skip invalid bboxes
    if bbox_width == 0 || bbox_height == 0 {
        anyhow::bail!("Invalid bbox dimensions: {}x{}", bbox_width, bbox_height);
    }
    
    // Extract the bbox region from the frame
    let expected_size = (bbox_width * bbox_height * 3) as usize;
    let mut cropped_data = Vec::with_capacity(expected_size);
    
    let frame_stride = (frame.width * 3) as usize;
    
    for y in y1..y2 {
        let row_offset = (y as usize) * frame_stride;
        let start_x = (x1 as usize) * 3;
        let end_x = (x2 as usize) * 3;
        
        let row_start = row_offset + start_x;
        let row_end = row_offset + end_x;
        
        cropped_data.extend_from_slice(&frame.data[row_start..row_end]);
    }
    
    // Verify cropped data size
    if cropped_data.len() != expected_size {
        anyhow::bail!(
            "Cropped data size mismatch: got {} bytes, expected {} ({}x{}x3)",
            cropped_data.len(),
            expected_size,
            bbox_width,
            bbox_height
        );
    }
    
    // Apply letterbox resize + padding + ImageNet normalization
    const TARGET_SIZE: u32 = 224;
    let preprocessed = processing::resize_letterbox_and_normalize_imagenet(
        &cropped_data,
        bbox_height,
        bbox_width,
        TARGET_SIZE,
        TARGET_SIZE,
        precision
    )
        .context("Error preprocessing bbox for DINOv3")?;
    
    Ok(preprocessed)
}

/// Performs operations on a given frame, including pre/post processing, inference on the given frame
pub async fn process_frame(
    inference_model: &InferenceModel,
    frame: Arc<RawFrame>
) -> Result<(FrameProcessStats, ResultEmbedding)> {
    let processing_start = Instant::now();

    // Pre process
    let measure_start = Instant::now();
    let precision = inference_model.model_config().precision;
    let frame_clone = Arc::clone(&frame);
    let pre_frame = tokio::task::spawn_blocking(move || {
        preprocess_frame(&frame_clone, precision)
    })
        .await
        .context("Preprocess task failed")?
        .context("Error preprocessing frame for DINOv3")?;
    let pre_proc_time = measure_start.elapsed();

    // Inference
    let measure_start = Instant::now();
    let raw_results = inference_model.infer(vec![pre_frame])
        .await
        .context("Error performing inference for DINOv3")?;
    let inference_time = measure_start.elapsed();

    // Post process
    let measure_start = Instant::now();
    let embeddings = tokio::task::spawn_blocking(move || {
        postprocess(raw_results, precision)
    })
        .await
        .context("Postprocess task failed")?
        .context("Error postprocessing frame for DINOv3")?;

    let embedding = match embeddings.into_iter().next() {
        Some(res) => res,
        None => anyhow::bail!("No inference results returned for DINOv3"),
    };

    let post_proc_time = measure_start.elapsed();

    // Statistics
    let mut stats = FrameProcessStats::default();
    stats.pre_processing = pre_proc_time.as_micros() as u64;
    stats.inference = inference_time.as_micros() as u64;
    stats.post_processing = post_proc_time.as_micros() as u64;
    stats.processing = processing_start.elapsed().as_micros() as u64;

    Ok((stats, embedding))
}

/// Performs operations on a given frame, including pre/post processing, inference on the given frame
pub async fn process_bboxes(
    inference_model: &InferenceModel,
    frame: Arc<RawFrame>,
    bboxes: Arc<Vec<ResultBBOX>>
) -> Result<(FrameProcessStats, Vec<ResultEmbedding>)> {
    let processing_start = Instant::now();

    // Pre process
    let measure_start = Instant::now();
    let precision = inference_model.model_config().precision;
    
    let tasks: Vec<_> = bboxes.iter().map(|bbox| {
        let frame_clone = Arc::clone(&frame);
        let bbox_clone = bbox.clone();
        
        tokio::task::spawn_blocking(move || {
            preprocess_bbox(&frame_clone, &bbox_clone, precision)
        })
    }).collect();

    let pre_bboxes_results = futures::future::try_join_all(tasks)
        .await
        .context("Preprocess tasks failed")?;
        
    let pre_bboxes: Vec<Vec<u8>> = pre_bboxes_results.into_iter()
        .collect::<Result<Vec<Vec<u8>>>>()
        .context("Error preprocessing bboxes for DINOv3")?;
        
    let pre_proc_time = measure_start.elapsed();

    // Inference
    let measure_start = Instant::now();
    let raw_results = inference_model.infer(pre_bboxes)
        .await
        .context("Error performing inference for DINOv3")?;
    let inference_time = measure_start.elapsed();

    // Post process
    let measure_start = Instant::now();
    let embeddings = tokio::task::spawn_blocking(move || {
        postprocess(raw_results, precision)
    })
        .await
        .context("Postprocess task failed")?
        .context("Error postprocessing bboxes for DINOv3")?;

    let post_proc_time = measure_start.elapsed();

    // Statistics
    let mut stats = FrameProcessStats::default();
    stats.pre_processing = pre_proc_time.as_micros() as u64;
    stats.inference = inference_time.as_micros() as u64;
    stats.post_processing = post_proc_time.as_micros() as u64;
    stats.processing = processing_start.elapsed().as_micros() as u64;

    Ok((stats, embeddings))
}