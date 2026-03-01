//! Responsible for handling video stream frames, sending them to inference
//! and populating results to third party systems

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::{OnceCell, RwLock, Semaphore};
use chrono::Utc;

// Custom modules
use crate::client_video::ClientVideo;
use crate::inference;
use crate::processing::{self, RawFrame, ResultBBOX, ResultEmbedding};
use crate::statistics::{FrameProcessStats, SourceStats};
use crate::utils::config::{AppConfig, InferenceModelType, InferenceTask, SourceConfig};
use crate::utils::queue::FixedSizeQueue;
use crate::utils::elastic::Elastic;

// Variables
pub static PROCESSORS: OnceCell<Arc<RwLock<HashMap<String, Arc<SourceProcessor>>>>> = OnceCell::const_new();
pub static MAX_QUEUE_FRAMES: usize = 15;

pub fn get_source_processors() -> Result<Arc<RwLock<HashMap<String, Arc<SourceProcessor>>>>> {
    PROCESSORS
        .get()
        .cloned()
        .context("Source processors not initiated")
}

/// Returns a source processor instance by given stream ID
pub async fn get_source_processor(stream_id: &str) -> Result<Arc<SourceProcessor>> {
    PROCESSORS
        .get()
        .context("Source processors not initiated")?
        .read()
        .await
        .get(stream_id)
        .cloned()
        .context("Error getting stream source processor")
}

/// Initiates source processors for given list of sources
pub async fn init_source_processors(app_config: &AppConfig) -> Result<()> {
    let processors = PROCESSORS
        .get_or_init(|| async { Arc::new(RwLock::new(HashMap::new())) })
        .await;
    let mut processors = processors.write().await;

    // Clean processors
    processors.clear();

    // Insert new processors
    for (source_id, source_config) in app_config.sources_config().sources.iter() {
        // Start processor
        let processor = Arc::new(SourceProcessor::new(
            source_id.to_string(),
            source_config.clone(),
            app_config.inference_config().task,
        ));

        processors.insert(source_id.to_string(), processor);
    }

    Ok(())
}

/// Responsible for managing inference/processing for each source
///
/// Performs inference for each source seperately. Allows us to control
/// each source seperately, with various settings, such as:
/// 1. confidence_threshold: What confidence threshold we apply to results for this specific source.
/// Especially relevant in case this source is known as more problematic and requires higher confidence
/// 2. inference_frame: How many frames we want to skip before performing inference. In other words,
/// "Inference on every N frame". This allows us to skip inference on frames when source has higher frame
/// rate, having minimal effect on the end user's experience.
#[allow(dead_code)]
pub struct SourceProcessor {
    // Settings for multi-threading
    queue: Arc<FixedSizeQueue<Arc<RawFrame>>>,
    queue_semaphore: Arc<Semaphore>,
    process_handle: tokio::task::JoinHandle<()>,

    // Source specific settings
    source_id: Arc<String>,
    source_config: Arc<SourceConfig>,
    source_stats: Arc<SourceStats>,
    inference_task: InferenceTask,
}

impl SourceProcessor {
    /// Creates a new instance of source processor
    ///
    /// 1. Creates a seperate channel of communication between the main thread and a seperate
    /// thread pool, so we can send frames for inference and not block the execution of other parts
    /// of our code.
    /// 2. Reports statistics about the given source processor in terms performance, including times of
    /// processing, how many successful/failed frames we have and what is our general success rate
    pub fn new(
        source_id: String,
        source_config: SourceConfig,
        inference_task: InferenceTask,
    ) -> Self {
        // Create global counters
        let source_id = Arc::new(source_id);
        let source_stats = Arc::new(SourceStats::new());
        let source_config = Arc::new(source_config);

        // Create a queue for frames. We set a maximum number of frames possible to be in queue at a given time
        // When the limit reaches, it drops the oldest frame in the queue, making it possible for new frames
        // to be added to the queue and be processed.
        let queue_stats = Arc::clone(&source_stats);
        let queue_drop_callback = move |_: Arc<RawFrame>| {
            queue_stats.frames_failed.fetch_add(1, Ordering::Relaxed);
        };
        let source_queue = Arc::new(FixedSizeQueue::<Arc<RawFrame>>::new(
            MAX_QUEUE_FRAMES,
            Some(queue_drop_callback),
        ));
        let queue_semaphore = Arc::new(Semaphore::new(MAX_QUEUE_FRAMES));

        // Create a seperate task for handling frames - performing inference
        let process_queue_semaphore = Arc::clone(&queue_semaphore);
        let process_source_queue = Arc::clone(&source_queue);
        let process_source_id = Arc::clone(&source_id);
        let process_source_config = Arc::clone(&source_config);
        let process_source_stats = Arc::clone(&source_stats);

        let process_handle = tokio::spawn(async move {
            let frame_process: Result<()> = async {
                loop {
                    // Try to acquire permit without blocking
                    match Arc::clone(&process_queue_semaphore).acquire_owned().await {
                        Ok(permit) => {
                            // Only pull from queue when we have a permit available
                            if let Some(frame) = process_source_queue.receiver.recv().await {
                                // Move values to the new thread
                                let process_source_id_ext = Arc::clone(&process_source_id);
                                let process_source_id_int = Arc::clone(&process_source_id);
                                let process_source_config = Arc::clone(&process_source_config);
                                let process_source_stats = Arc::clone(&process_source_stats);
                                let process_frame = Arc::clone(&frame);

                                // Spawn processing in a new thread with permit
                                tokio::spawn(async move {
                                    // Keep permit alive until processing completes
                                    let _permit = permit;

                                    let process_result = SourceProcessor::process_frame_internal(
                                        process_source_id_int,
                                        &process_source_config,
                                        process_frame,
                                        inference_task,
                                    )
                                    .await;

                                    // Count processing statistics
                                    process_source_stats
                                        .frames_total
                                        .fetch_add(1, Ordering::Relaxed);
                                    process_source_stats
                                        .frames_expected
                                        .fetch_add(1, Ordering::Relaxed);
                                    match &process_result {
                                        Ok(stats) => {
                                            process_source_stats
                                                .frames_success
                                                .fetch_add(1, Ordering::Relaxed);

                                            // Add inference statistics to counters
                                            process_source_stats.accumulate(&stats);
                                        }
                                        Err(_) => {
                                            process_source_stats
                                                .frames_failed
                                                .fetch_add(1, Ordering::Relaxed);
                                        }
                                    }

                                    // Handle processing error
                                    if let Err(e) = process_result {
                                        tracing::error!(
                                            source_id = &*process_source_id_ext,
                                            error = ?e,
                                            "error processing source frame"
                                        )
                                    };
                                });
                            }
                        }
                        Err(e) => {
                            tracing::info!(
                                source_id = &*process_source_id,
                                error = e.to_string(),
                                "Error acquiring permit for parallelism. Should not happen"
                            )
                        }
                    }
                }
            }
            .await;

            if let Err(e) = frame_process {
                tracing::error!(
                    source_id = &*process_source_id,
                    error = e.to_string(),
                    "Stopped processing frames - due to fatal error"
                )
            }
        });

        tracing::info!(source_id = &*source_id, "initiated client processing");

        Self {
            queue: source_queue,
            queue_semaphore,
            process_handle,
            source_id,
            source_config,
            source_stats,
            inference_task,
        }
    }

    /// Sends inference requests to a seperate thread pool
    pub async fn process_frame(&self, raw_frame: Vec<u8>, height: u32, width: u32, pts: u64) {
        let frames_total = self.source_stats.frames_total.load(Ordering::Relaxed);

        // Send inference results on every N frame
        if (frames_total + 1) % (self.source_config.inf_frame as u64) == 0 {
            // Create new frame object
            let frame = Arc::new(RawFrame {
                data: raw_frame,
                height,
                width,
                pts,
                added: tokio::time::Instant::now(),
            });

            // Send new frame to queue
            self.queue.sender.send_async(frame).await;
        } else {
            // Add to statistics
            self.source_stats
                .frames_total
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Used to perform inference on a raw frame and return stats about timing
    #[allow(unreachable_patterns)]
    async fn process_frame_internal(
        source_id: Arc<String>,
        source_config: &SourceConfig,
        frame: Arc<RawFrame>,
        inference_task: InferenceTask,
    ) -> Result<FrameProcessStats> {
        let frame_queue_time = frame.added.elapsed();

        // Perform inference on raw frame and populate results
        let mut stats = match inference_task {
            InferenceTask::ObjectDetection => {
                // Get BBOXes for frame
                let yolo_model = inference::get_inference_model(InferenceModelType::YOLO)?;
                let yolo_frame = Arc::clone(&frame);
                let (mut yolo_stats, bboxes) =
                    processing::yolo::process_frame(&yolo_model, &source_config, yolo_frame)
                        .await?;

                // Populate BBOXes if we have any
                if bboxes.len() > 0 {
                    let measure_start = tokio::time::Instant::now();

                    // Populate BBOXes to third party services
                    let results_source_id = Arc::clone(&source_id);
                    let results_frame = Arc::clone(&frame);
                    let results_arc = Arc::new(bboxes);
                    SourceProcessor::populate_bboxes(results_source_id, results_frame, results_arc)
                        .await;

                    // Update results time
                    let results_time = measure_start.elapsed();
                    yolo_stats.results += results_time.as_micros() as u64;
                }

                yolo_stats
            }
            InferenceTask::Embedding => {
                let mut final_stats = FrameProcessStats::default();

                // Prepare models
                let yolo_model = inference::get_inference_model(InferenceModelType::YOLO)?;
                let dino_model = inference::get_inference_model(InferenceModelType::DINO)?;
                
                // Prepare inputs for parallel execution
                let yolo_frame = Arc::clone(&frame);
                let dino_frame = Arc::clone(&frame);
                
                // Execute YOLO and DINO frame inference in parallel
                let (
                    (yolo_stats, bboxes),
                    (dino_frame_stats, dino_frame_embedding)
                ) = tokio::try_join!(
                    processing::yolo::process_frame(yolo_model, &source_config, yolo_frame),
                    processing::dino::process_frame(dino_model, dino_frame)
                )?;

                // Accumulate statistics
                final_stats.accumulate(&yolo_stats);
                final_stats.accumulate(&dino_frame_stats);
                
                // Process BBOXes with DINO
                let mut all_embeddings = Vec::with_capacity(1 + bboxes.len());
                all_embeddings.push(dino_frame_embedding);

                if bboxes.len() > 0 {
                    let bboxes = Arc::new(bboxes);

                    let dino_bbox_model = inference::get_inference_model(InferenceModelType::DINO_OBJECTS)?;
                    let dino_bbox_frame = Arc::clone(&frame);
                    let dino_bbox_bboxes = Arc::clone(&bboxes);
                
                    let (dino_bbox_stats, mut dino_bbox_embeddings) = processing::dino::process_bboxes(
                        dino_bbox_model,
                        dino_bbox_frame,
                        dino_bbox_bboxes
                    ).await?;

                    all_embeddings.append(&mut dino_bbox_embeddings);
                    final_stats.accumulate(&dino_bbox_stats);
                }

                // Populate embeddings if we have any
                if all_embeddings.len() > 0 {
                    let all_embeddings = Arc::new(all_embeddings);
                    let measure_start = tokio::time::Instant::now();

                    // Populate embeddings to third party services
                    let results_source_id = Arc::clone(&source_id);
                    let results_frame = Arc::clone(&frame);
                    let results_embeddings = Arc::clone(&all_embeddings);
                    SourceProcessor::populate_embeddings(
                        results_source_id,
                        results_frame,
                        results_embeddings,
                    )
                    .await;

                    // Update results time
                    let results_time = measure_start.elapsed();
                    final_stats.results += results_time.as_micros() as u64;
                }

                final_stats
            }
            _ => anyhow::bail!("Model task is not supported for processing!"),
        };

        // Return statistics
        stats.queue = frame_queue_time.as_micros() as u64;
        stats.processing += frame_queue_time.as_micros() as u64;
        Ok(stats)
    }

    /// Populates BBOXes to third party services
    pub async fn populate_bboxes(
        source_id: Arc<String>,
        frame: Arc<RawFrame>,
        bboxes: Arc<Vec<ResultBBOX>>,
    ) {
        let bboxes = Arc::new(bboxes);

        // Send to client video
        let client_source_id = Arc::clone(&source_id);
        let client_frame = Arc::clone(&frame);
        let client_bboxes = Arc::clone(&bboxes);

        if let Err(e) = ClientVideo::populate_bboxes(
            &client_source_id, 
            &client_frame, 
            &client_bboxes
        ).await {
            tracing::warn!(
                source_id = &*source_id,
                error = ?e,
                "Failed to populate bboxes to client video"
            );
        };
    }

    /// Populates embedding to third party services
    #[allow(unused_variables)]
    pub async fn populate_embeddings(
        source_id: Arc<String>,
        frame: Arc<RawFrame>,
        embeddings: Arc<Vec<ResultEmbedding>>,
    ) {
        if let Err(e) = Elastic::populate_embeddings(
            &source_id, 
            Utc::now().timestamp_millis(), 
            &embeddings
        ).await {
            tracing::warn!(
                source_id = &*source_id,
                error = ?e,
                "Failed to populate embeddings to elastic"
            );
        }
    }
}

impl SourceProcessor {
    pub fn source_stats(&self) -> &SourceStats {
        &self.source_stats
    }
}

impl Drop for SourceProcessor {
    fn drop(&mut self) {
        // Abort tokio tasks
        self.process_handle.abort();
    }
}
