use anyhow::{Result, Context};
use std::time::Instant;
use std::sync::Arc;

// Custom modules
use crate::inference::InferenceModel;
use crate::statistics::FrameProcessStats;
use crate::processing::{self, RawFrame, ResultBBOX};
use crate::utils::config::SourceConfig;
use crate::utils::config::InferencePrecision;

/// Performs pre-processing on raw RGB frame for YOLO models
/// 
/// Performs the following steps of processing:
/// 1. Resizes the given image to 640x640 while preserving aspect ratio.
/// Applying letterbox padding to complete the missing pixels for certain aspect ratios.
/// 2. Normalizes pixels from 0-255 to 0-1
/// 3. Converting raw pixel values to required precision datatype
/// 4. Outputs raw bytes ordered by color channels(Planar): \[RRRBBBGGG\]
pub fn preprocess(
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

    // Preprocess with letterbox resize + YOLO normalization
    const TARGET_SIZE: u32 = 640;
    processing::resize_letterbox_and_normalize(
        &frame.data,
        frame.height,
        frame.width,
        TARGET_SIZE,
        TARGET_SIZE,
        precision
    )
}

/// Perform NMS reduction of bboxes
#[inline(never)] // Don't inline to keep instruction cache hot for main loop
fn bbox_nms(detections: &mut Vec<ResultBBOX>, nms_threshold: f32) {
    let len = detections.len();
    if len <= 1 {
        return;
    }
    
    // Sort in-place by score descending
    detections.sort_unstable_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let mut write_idx = 0;
    
    for i in 0..len {
        let detection_i = unsafe { *detections.get_unchecked(i) };
        let mut should_keep = true;
        
        // Check against already kept detections
        for j in 0..write_idx {
            let kept = unsafe { detections.get_unchecked(j) };
            
            // Skip different classes
            if kept.class != detection_i.class {
                continue;
            }
            
            // Compute IoU inline
            let x1_max = detection_i.bbox[0].max(kept.bbox[0]);
            let y1_max = detection_i.bbox[1].max(kept.bbox[1]);
            let x2_min = detection_i.bbox[2].min(kept.bbox[2]);
            let y2_min = detection_i.bbox[3].min(kept.bbox[3]);
            
            // Check for intersection
            if x1_max < x2_min && y1_max < y2_min {
                let intersection = (x2_min - x1_max) * (y2_min - y1_max);
                let area_i = (detection_i.bbox[2] - detection_i.bbox[0]) * (detection_i.bbox[3] - detection_i.bbox[1]);
                let area_j = (kept.bbox[2] - kept.bbox[0]) * (kept.bbox[3] - kept.bbox[1]);
                let union = area_i + area_j - intersection;
                
                if intersection > nms_threshold * union {
                    should_keep = false;
                    break;
                }
            }
        }
        
        if should_keep {
            unsafe {
                *detections.get_unchecked_mut(write_idx) = detection_i;
            }
            write_idx += 1;
        }
    }
    
    detections.truncate(write_idx);
}

/// Performs post-processing on inference results for YOLO models
/// 
/// Including the following steps of processing:
/// 1. Convert BBOX coordinates from (x, y, w, h) to (x1, y1, x2, y2) together
/// with restoring the letterbox padding applied during pre-processing
/// 2. Finds out the class id with the max probability - making it the 
/// class for the bbox along with its probabiliy
/// 3. Filter BBOXes on a given confidence threshold, before applying NMS(boosts performance significantly)
/// 4. Perform NMS on left over BBOXes
pub fn postprocess(
    results: &[u8],
    original_frame: &RawFrame,
    output_shape: &[i64],
    precision: InferencePrecision,
    pred_conf_threshold: f32,
    nms_iou_threshold: f32,
) -> Result<Vec<ResultBBOX>> {
    // Validate model output shape
    if output_shape.len() != 2 {
        anyhow::bail!(
            format!(
                "Got unexpected size of model output shape. Got {}, expected 2",
                output_shape.len()
            )
        );
    }

    let target_features = output_shape[0] as u32;
    let target_anchors = output_shape[1] as u32;
    let target_classes = target_features - 4;
    
    // Validate size of output data
    let expected_size = match precision {
        InferencePrecision::FP16 => target_anchors * target_features * 2,
        InferencePrecision::FP32 => target_anchors * target_features * 4,
    } as usize;
    
    if results.len() != expected_size {
        anyhow::bail!(
            format!(
                "Got unexpected size of model output data ({}). Got {}, expected {}",
                precision.to_string(),
                results.len(),
                expected_size
            )
        );
    }
    
    // Precompute letterbox parameters
    const TARGET_SIZE: u32 = 640;
    let letterbox = processing::calculate_letterbox(
        original_frame.height, 
        original_frame.width, 
        TARGET_SIZE
    );
    
    // Pre-allocate with exact capacity estimate (typically ~100-200 detections)
    let mut detections = Vec::with_capacity(256);
    
    match precision {
        InferencePrecision::FP16 => {
            let u16_data = unsafe {
                std::slice::from_raw_parts(results.as_ptr() as *const u16, results.len() / 2)
            };
            
            // Precompute strides
            let stride1 = target_anchors;
            let stride2 = target_anchors * 2;
            let stride3 = target_anchors * 3;
            let stride4 = target_anchors * 4;
            
            // Process anchors with optimized memory access pattern
            for anchor_idx in 0..target_anchors {
                unsafe {
                    // Load all bbox values at once for better cache usage
                    let x = processing::get_f16_to_f32_lut(*u16_data.get_unchecked(anchor_idx as usize));
                    let y = processing::get_f16_to_f32_lut(*u16_data.get_unchecked((stride1 + anchor_idx) as usize));
                    let w = processing::get_f16_to_f32_lut(*u16_data.get_unchecked((stride2 + anchor_idx) as usize));
                    let h = processing::get_f16_to_f32_lut(*u16_data.get_unchecked((stride3 + anchor_idx) as usize));
                    
                    // Fused bbox transformation
                    let half_w = w * 0.5;
                    let half_h = h * 0.5;
                    let x1 = (x - half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y1 = (y - half_h - letterbox.pad_y as f32) * letterbox.inv_scale;
                    let x2 = (x + half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y2 = (y + half_h - letterbox.pad_y as f32) * letterbox.inv_scale;
                    
                    // Find max class with unrolled loop for common cases
                    let mut max_score: f32 = 0.0;
                    let mut max_class: u32 = 0;
                    
                    let class_base = stride4 + anchor_idx;
                    
                    for class_idx in 0..target_classes {
                        let prob_idx = (class_base + class_idx * stride1) as usize;
                        let score = processing::get_f16_to_f32_lut(*u16_data.get_unchecked(prob_idx));
                        if score > max_score {
                            max_score = score;
                            max_class = class_idx;
                        }
                    }
                    
                    // Only store if above threshold
                    if max_score >= pred_conf_threshold {
                        detections.push(
                            ResultBBOX {
                                bbox: [x1, y1, x2, y2],
                                class: max_class,
                                score: max_score,
                            }
                        );
                    }
                }
            }
        }
        InferencePrecision::FP32 => {
            let f32_data = unsafe {
                std::slice::from_raw_parts(results.as_ptr() as *const f32, results.len() / 4)
            };
            
            // Precompute strides
            let stride1 = target_anchors;
            let stride2 = target_anchors * 2;
            let stride3 = target_anchors * 3;
            let stride4 = target_anchors * 4;
            
            for anchor_idx in 0..target_anchors {
                unsafe {
                    // Load bbox values
                    let x = *f32_data.get_unchecked(anchor_idx as usize);
                    let y = *f32_data.get_unchecked((stride1 + anchor_idx) as usize);
                    let w = *f32_data.get_unchecked((stride2 + anchor_idx) as usize);
                    let h = *f32_data.get_unchecked((stride3 + anchor_idx) as usize);
                    
                    // Fused bbox transformation
                    let half_w = w * 0.5;
                    let half_h = h * 0.5;
                    let x1 = (x - half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y1 = (y - half_h - letterbox.pad_y as f32) * letterbox.inv_scale;
                    let x2 = (x + half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y2 = (y + half_h - letterbox.pad_y as f32) * letterbox.inv_scale;
                    
                    // Find max class with unrolling
                    let mut max_score: f32 = 0.0;
                    let mut max_class: u32 = 0;
                    
                    let class_base = stride4 + anchor_idx;
                    
                    for class_idx in 0..target_classes {
                        let prob_idx = (class_base + class_idx * stride1) as usize;
                        let score = *f32_data.get_unchecked(prob_idx);
                        if score > max_score {
                            max_score = score;
                            max_class = class_idx;
                        }
                    }
                    
                    if max_score >= pred_conf_threshold {
                        detections.push(
                            ResultBBOX {
                                bbox: [x1, y1, x2, y2],
                                class: max_class,
                                score: max_score,
                            }
                        );
                    }
                }
            }
        }
    }
    
    // Fast NMS only if needed
    if detections.len() > 1 {
        bbox_nms(&mut detections, nms_iou_threshold);
    }
    
    Ok(detections)
}

/// Performs operations on a given frame, including pre/post processing, inference on the given frame
pub async fn process_frame(
    inference_model: &InferenceModel, 
    source_config: &SourceConfig,
    frame: Arc<RawFrame>
) -> Result<(FrameProcessStats, Vec<ResultBBOX>)> {
    let processing_start = Instant::now();

    // Pre process
    let measure_start = Instant::now();
    let precision = inference_model.model_config().precision;
    let frame_clone = Arc::clone(&frame);
    let pre_frame = tokio::task::spawn_blocking(move || {
        preprocess(&frame_clone, precision)
    })
        .await
        .context("Preprocess task failed")?
        .context("Error preprocessing image for YOLO")?;
    let pre_proc_time = measure_start.elapsed();

    // Inference
    let measure_start = Instant::now();
    let raw_results = inference_model.infer(vec![pre_frame])
        .await
        .context("Error performing inference for YOLO")?;
    let inference_time = measure_start.elapsed();

    let raw_results = match raw_results.into_iter().next() {
        Some(res) => res,
        None => anyhow::bail!("No inference results returned for YOLO"),
    };

    // Post process
    let measure_start = Instant::now();
    let post_output_shape = inference_model.model_config().output_shape.clone();
    let post_conf_threshold = source_config.conf_threshold;
    let post_nms_iou_threshold = source_config.nms_iou_threshold;
    
    let bboxes = tokio::task::spawn_blocking(move || {
        postprocess(
            &raw_results, 
            &frame,
            &post_output_shape,
            precision,
            post_conf_threshold,
            post_nms_iou_threshold
        )
    })
        .await
        .context("Postprocess task failed")?
        .context("Error postprocessing BBOXes for YOLO")?;
    let post_proc_time = measure_start.elapsed();

    // Statistics
    let mut stats = FrameProcessStats::default();
    stats.pre_processing = pre_proc_time.as_micros() as u64;
    stats.inference = inference_time.as_micros() as u64;
    stats.post_processing = post_proc_time.as_micros() as u64;
    stats.processing = processing_start.elapsed().as_micros() as u64;

    Ok((stats, bboxes))
}