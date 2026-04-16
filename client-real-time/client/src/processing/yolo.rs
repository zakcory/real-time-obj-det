use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Instant;

// Custom modules
use crate::inference::InferenceModel;
use crate::processing::{self, RawFrame, ResultBBOX};
use crate::statistics::FrameProcessStats;
use crate::utils::config::{InferencePrecision, ModelOutputConfig, SourceConfig};

/// Performs pre-processing on raw RGB frame for YOLO models
///
/// Performs the following steps of processing:
/// 1. Resizes the given image to 640x640 while preserving aspect ratio.
/// Applying letterbox padding to complete the missing pixels for certain aspect ratios.
/// 2. Normalizes pixels from 0-255 to 0-1
/// 3. Converting raw pixel values to required precision datatype
/// 4. Outputs raw bytes ordered by color channels(Planar): \[RRRBBBGGG\]
pub fn preprocess_frame(
    frame: &RawFrame,
    precision: InferencePrecision,
    target_size: u32,
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
    processing::resize_letterbox_and_normalize(
        &frame.data,
        frame.height,
        frame.width,
        target_size,
        target_size,
        precision,
    )
}

/// Perform NMS reduction of bboxes
#[inline(never)]
fn bbox_nms(detections: &mut Vec<ResultBBOX>, nms_threshold: f32) {
    let len = detections.len();
    if len <= 1 {
        return;
    }

    detections.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut write_idx = 0;

    for i in 0..len {
        let detection_i = unsafe { *detections.get_unchecked(i) };
        let mut should_keep = true;

        for j in 0..write_idx {
            let kept = unsafe { detections.get_unchecked(j) };

            if kept.class != detection_i.class {
                continue;
            }

            let x1_max = detection_i.bbox[0].max(kept.bbox[0]);
            let y1_max = detection_i.bbox[1].max(kept.bbox[1]);
            let x2_min = detection_i.bbox[2].min(kept.bbox[2]);
            let y2_min = detection_i.bbox[3].min(kept.bbox[3]);

            if x1_max < x2_min && y1_max < y2_min {
                let intersection = (x2_min - x1_max) * (y2_min - y1_max);
                let area_i = (detection_i.bbox[2] - detection_i.bbox[0])
                    * (detection_i.bbox[3] - detection_i.bbox[1]);
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

fn postprocess_enms(
    raw_results: &[u8],
    outputs: &[ModelOutputConfig],
    original_frame: &RawFrame,
    target_size: u32,
    pred_conf_threshold: f32,
) -> Result<Vec<ResultBBOX>> {
    if outputs.len() != 1 {
        anyhow::bail!(
            "Packed NMS mode expects single output, model has {} outputs",
            outputs.len()
        );
    }
    let packed_output = outputs
        .first()
        .context("Missing packed NMS output config")?;
    if packed_output.data_type != InferencePrecision::FP32 {
        anyhow::bail!(
            "Packed EfficientNMS expects FP32 output, got {}",
            packed_output.data_type.to_string()
        );
    }
    if raw_results.len() % 4 != 0 {
        anyhow::bail!("Invalid packed FP32 output length: {}", raw_results.len());
    }

    let packed = unsafe {
        std::slice::from_raw_parts(raw_results.as_ptr() as *const f32, raw_results.len() / 4)
    };
    if (packed.len() - 1) % 6 != 0 {
        anyhow::bail!(
            "Invalid packed EfficientNMS output length {} (expected 1 + 6*K)",
            packed.len()
        );
    }

    // Refer to the .md made in nexus-optimizing
    let max_boxes = (packed.len() - 1) / 6;
    let valid_detections = packed[0].round().max(0.0) as usize;
    let boxes_start = 1;
    let scores_start = 1 + (4 * max_boxes);
    let classes_start = 1 + (5 * max_boxes);
    let detections_to_read = valid_detections.min(max_boxes);

    let letterbox =
        processing::calculate_letterbox(original_frame.height, original_frame.width, target_size);
    let frame_max_x = (original_frame.width.saturating_sub(1)) as f32;
    let frame_max_y = (original_frame.height.saturating_sub(1)) as f32;
    let mut detections = Vec::with_capacity(detections_to_read);

    for i in 0..detections_to_read {
        let score = packed[scores_start + i];
        if score < pred_conf_threshold {
            continue;
        }

        let base = boxes_start + (i * 4);
        let y1_input = packed[base];
        let x1_input = packed[base + 1];
        let y2_input = packed[base + 2];
        let x2_input = packed[base + 3];

        let x1 =
            ((x1_input - letterbox.pad_x as f32) * letterbox.inv_scale).clamp(0.0, frame_max_x);
        let y1 =
            ((y1_input - letterbox.pad_y as f32) * letterbox.inv_scale).clamp(0.0, frame_max_y);
        let x2 =
            ((x2_input - letterbox.pad_x as f32) * letterbox.inv_scale).clamp(0.0, frame_max_x);
        let y2 =
            ((y2_input - letterbox.pad_y as f32) * letterbox.inv_scale).clamp(0.0, frame_max_y);

        detections.push(ResultBBOX {
            bbox: [x1, y1, x2, y2],
            class: packed[classes_start + i].round().max(0.0) as u32,
            score,
        });
    }

    Ok(detections)
}

fn postprocess_raw_frame(
    raw_results: &[u8],
    outputs: &[ModelOutputConfig],
    original_frame: &RawFrame,
    target_size: u32,
    precision: InferencePrecision,
    pred_conf_threshold: f32,
    nms_iou_threshold: f32,
) -> Result<Vec<ResultBBOX>> {
    let target_features = outputs
        .first()
        .and_then(|o| o.shape.first())
        .copied()
        .ok_or(anyhow::anyhow!("Invalid output shape for raw YOLO model"))?
        as u32;
    let target_anchors = 8400_u32;
    let target_classes = target_features - 4;
    let expected_size = match precision {
        InferencePrecision::FP16 => target_anchors * target_features * 2,
        InferencePrecision::FP32 => target_anchors * target_features * 4,
    } as usize;

    if raw_results.len() != expected_size {
        anyhow::bail!(
            "Got unexpected size of model output data ({}). Got {}, expected {}",
            precision.to_string(),
            raw_results.len(),
            expected_size
        );
    }

    let letterbox =
        processing::calculate_letterbox(original_frame.height, original_frame.width, target_size);
    let mut detections = Vec::with_capacity(256);

    match precision {
        InferencePrecision::FP16 => {
            let u16_data = unsafe {
                std::slice::from_raw_parts(
                    raw_results.as_ptr() as *const u16,
                    raw_results.len() / 2,
                )
            };

            let stride1 = target_anchors;
            let stride2 = target_anchors * 2;
            let stride3 = target_anchors * 3;
            let stride4 = target_anchors * 4;

            for anchor_idx in 0..target_anchors {
                unsafe {
                    let x = processing::get_f16_to_f32_lut(
                        *u16_data.get_unchecked(anchor_idx as usize),
                    );
                    let y = processing::get_f16_to_f32_lut(
                        *u16_data.get_unchecked((stride1 + anchor_idx) as usize),
                    );
                    let w = processing::get_f16_to_f32_lut(
                        *u16_data.get_unchecked((stride2 + anchor_idx) as usize),
                    );
                    let h = processing::get_f16_to_f32_lut(
                        *u16_data.get_unchecked((stride3 + anchor_idx) as usize),
                    );

                    let half_w = w * 0.5;
                    let half_h = h * 0.5;
                    let x1 = (x - half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y1 = (y - half_h - letterbox.pad_y as f32) * letterbox.inv_scale;
                    let x2 = (x + half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y2 = (y + half_h - letterbox.pad_y as f32) * letterbox.inv_scale;

                    let mut max_score: f32 = 0.0;
                    let mut max_class: u32 = 0;
                    let class_base = stride4 + anchor_idx;

                    for class_idx in 0..target_classes {
                        let prob_idx = (class_base + class_idx * stride1) as usize;
                        let score =
                            processing::get_f16_to_f32_lut(*u16_data.get_unchecked(prob_idx));
                        if score > max_score {
                            max_score = score;
                            max_class = class_idx;
                        }
                    }

                    if max_score >= pred_conf_threshold {
                        detections.push(ResultBBOX {
                            bbox: [x1, y1, x2, y2],
                            class: max_class,
                            score: max_score,
                        });
                    }
                }
            }
        }
        InferencePrecision::FP32 => {
            let f32_data = unsafe {
                std::slice::from_raw_parts(
                    raw_results.as_ptr() as *const f32,
                    raw_results.len() / 4,
                )
            };

            let stride1 = target_anchors;
            let stride2 = target_anchors * 2;
            let stride3 = target_anchors * 3;
            let stride4 = target_anchors * 4;

            for anchor_idx in 0..target_anchors {
                unsafe {
                    let x = *f32_data.get_unchecked(anchor_idx as usize);
                    let y = *f32_data.get_unchecked((stride1 + anchor_idx) as usize);
                    let w = *f32_data.get_unchecked((stride2 + anchor_idx) as usize);
                    let h = *f32_data.get_unchecked((stride3 + anchor_idx) as usize);

                    let half_w = w * 0.5;
                    let half_h = h * 0.5;
                    let x1 = (x - half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y1 = (y - half_h - letterbox.pad_y as f32) * letterbox.inv_scale;
                    let x2 = (x + half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y2 = (y + half_h - letterbox.pad_y as f32) * letterbox.inv_scale;

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
                        detections.push(ResultBBOX {
                            bbox: [x1, y1, x2, y2],
                            class: max_class,
                            score: max_score,
                        });
                    }
                }
            }
        }
    }

    if detections.len() > 1 {
        bbox_nms(&mut detections, nms_iou_threshold);
    }

    Ok(detections)
}

/// Performs operations on a given frame, including pre/post processing, inference on the given frame
pub async fn process_frame(
    inference_model: &InferenceModel,
    source_config: &SourceConfig,
    frame: Arc<RawFrame>,
) -> Result<(FrameProcessStats, Vec<ResultBBOX>)> {
    let processing_start = Instant::now();
    let target_size = inference_model
        .model_config()
        .input_shape
        .last()
        .ok_or(anyhow::anyhow!("Invalid input shape"))?
        .clone() as u32;

    // Pre process
    let measure_start = Instant::now();
    let frame_clone = Arc::clone(&frame);
    let precision = inference_model.model_config().precision;
    let pre_frame =
        tokio::task::spawn_blocking(move || preprocess_frame(&frame_clone, precision, target_size))
            .await
            .context("Preprocess task failed")?
            .context("Error preprocessing image for YOLO")?;
    let pre_proc_time = measure_start.elapsed();

    let post_conf_threshold = source_config.conf_threshold;
    let post_nms_iou_threshold = source_config.nms_iou_threshold;
    let nms_in_triton = inference_model.model_config().nms_in_triton();
    let outputs = inference_model
        .model_config()
        .resolved_outputs()
        .context("Invalid model output configuration for YOLO")?;

    let inference_time;
    let post_proc_time;
    let measure_start = Instant::now();
    let raw_results = inference_model
        .infer(vec![pre_frame])
        .await
        .context("Error performing inference for YOLO")?;
    inference_time = measure_start.elapsed();

    let raw_results = match raw_results.into_iter().next() {
        Some(res) => res,
        None => anyhow::bail!("No inference results returned for YOLO"),
    };

    let measure_start = Instant::now();
    let bboxes: Vec<ResultBBOX> = tokio::task::spawn_blocking(move || {
        if nms_in_triton {
            postprocess_enms(
                &raw_results,
                &outputs,
                &frame,
                target_size,
                post_conf_threshold,
            )
        } else {
            postprocess_raw_frame(
                &raw_results,
                &outputs,
                &frame,
                target_size,
                precision,
                post_conf_threshold,
                post_nms_iou_threshold,
            )
        }
    })
    .await
    .context("Postprocess task failed")?
    .context("Error postprocessing YOLO output")?;
    post_proc_time = measure_start.elapsed();

    // Statistics
    let mut stats = FrameProcessStats::default();
    stats.pre_processing = pre_proc_time.as_micros() as u64;
    stats.inference = inference_time.as_micros() as u64;
    stats.post_processing = post_proc_time.as_micros() as u64;
    stats.processing = processing_start.elapsed().as_micros() as u64;

    Ok((stats, bboxes))
}
