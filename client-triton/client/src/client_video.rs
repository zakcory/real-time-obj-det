use anyhow::{Result, Context};
use libloading::{Library, Symbol};
use libc::{c_int, c_ulonglong, c_char, c_void};
use std::slice;
use std::sync::Arc;
use serde_json::json;
use std::ffi::CString;
use std::sync::OnceLock;

// Custom modules
use crate::source;
use crate::utils::config::AppConfig;
use crate::processing::{RawFrame, ResultBBOX};

/// Client as static global variable
pub static CLIENT_VIDEO: OnceLock<Arc<ClientVideo>> = OnceLock::new();

pub fn get_client_video() -> Result<Arc<ClientVideo>> {
    let client_video = CLIENT_VIDEO.get()
        .context("Error creating client video")?;

    Ok(Arc::clone(client_video))
}

pub fn init_client_video() -> Result<()> {
    let client_video = ClientVideo::new()
        .context("Error creating client video")?;

    CLIENT_VIDEO.set(Arc::new(client_video))
        .map_err(|_| anyhow::anyhow!("Error setting up client video"))?;

    Ok(())
}

// C Types
pub type SourceFramesCb = extern "C" fn(source_id: c_int, frame: *const u8, width: c_int, height: c_int, pts: c_ulonglong);
pub type SourceStoppedCb = extern "C" fn(source_id: c_int);
pub type SourceNameCb = extern "C" fn(source_id: c_int, source_name: *const c_char);
pub type SourceStatusCb = extern "C" fn(source_id: c_int, source_status: c_int);
pub type InitMultipleSourcesFn = extern "C" fn(source_ids: *const c_int, size: c_int, log_level: c_int);
pub type PostResultsFn = extern "C" fn(source_id: c_int, result_json: *const c_char) -> c_int;
pub type FreeCPtrFn = extern "C" fn(ptr: *const c_void);
pub type SetCallbacksFn = extern "C" fn(
    source_frames: SourceFramesCb,
    source_stopped: SourceStoppedCb,
    source_name: SourceNameCb,
    source_status: SourceStatusCb
);

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogLevel {
    Regular = 0,
    Debug = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum SourceStatus {
    Ok = 0,
    NotStreaming = 1,
    NotFound = 2,
    ConnectionError = 3,
    DecodeError = 4,
}

pub struct ClientVideo {
    library: Library
}

impl ClientVideo {
    pub fn new() -> Result<Self> {
        // Load dynamic library
        let library = unsafe {
            Library::new("secrets/libclient_video.so")?
        };

        Ok(
            Self {
                library
            }
        )
    }

    pub async fn init_sources(app_config: &AppConfig) -> Result<()> {
        // Set callbacks
        ClientVideo::set_callbacks().await
            .context("Error setting Client Video callbacks")?;

        // Start sources
        let source_ids: Vec<c_int> = app_config.sources_config().sources
            .keys()
            .filter_map(|k| k.parse::<c_int>().ok())
            .collect();
        ClientVideo::start_sources(source_ids).await
            .context("Error starting Client Video sources")?;
        
        Ok(())
    }

    // Library function
    async fn set_callbacks() -> Result<()> {
        let client_video = get_client_video()?;

        tokio::task::spawn_blocking(move || -> Result<()> {
            unsafe {
                let lib_set_callbacks: Symbol<SetCallbacksFn> = client_video.library()
                    .get(b"SetCallbacks")
                    .context("Cannot get 'SetCallbacks' function")?;


                lib_set_callbacks(
                    ClientVideo::_source_frames_callback,
                    ClientVideo::_source_stopped_callback,
                    ClientVideo::_source_name_callback,
                    ClientVideo::_source_status_callback
                )
            }

            Ok(())
        }).await
            .context("Error trying to set callbacks in video client")?
            .context("Error setting callbacks in video client")?;

        Ok(())
    }

    async fn start_sources(source_ids: Vec<c_int>) -> Result<()> {
        let client_video = get_client_video()?;

        if source_ids.len() == 0 {
            anyhow::bail!("No valid sources are avaliable");
        }

        tokio::task::spawn_blocking(move || -> Result<()> {
            unsafe {
                let lib_init_multiple_sources: Symbol<InitMultipleSourcesFn> = client_video.library()
                    .get(b"InitMultipleSources")
                    .context("Cannot get 'InitMultipleSources' function")?;


                lib_init_multiple_sources(
                    source_ids.as_ptr() as *const c_int, 
                    source_ids.len() as c_int, 
                    LogLevel::Regular as c_int
                )
            }

            Ok(())
        }).await
            .context("Error trying to initiate sources in video client")?
            .context("Error initiating source in video client")?;

        Ok(())
    }
    
    pub async fn populate_bboxes(source_id: &str, frame: &RawFrame, bboxes: &[ResultBBOX]) -> Result<()> {
        // Format BBOXes output for sending it back to the client
        let bboxes_json: Vec<_> = bboxes
            .iter()
            .map(|bbox| {
                // Get bbox corners - indexes of pixels in frame, as if it was a 1d array
                let (top_left_corner, bottom_right_corner) = bbox.corners_coordinates(frame);

                json!({
                    "pts": frame.pts,
                    "top_left_corner": top_left_corner,
                    "bottom_right_corner": bottom_right_corner,
                    "class_name": bbox.class_name(),
                    "confidence": bbox.score
                })
            })
            .collect();
        
        let bboxes_result_json = json!({
            "stream_id": source_id,
            "bboxes": bboxes_json
        }).to_string();


        // Send back to client
        let client_video = get_client_video()?;
        let results_bboxes = CString::new(bboxes_result_json)
            .context("Error converting bboxes to C string")?;
        let results_source_id = source_id.parse::<c_int>()
            .expect("Failed to convert source id to integer");

        tokio::task::spawn_blocking(move || -> Result<()> {
            unsafe {
                let lib_post_results: Symbol<PostResultsFn> = client_video.library()
                    .get(b"PostResults")
                    .context("Cannot get 'PostResults' function")?;


                let result = lib_post_results(
                    results_source_id,
                    results_bboxes.into_raw()
                );

                // Check whether posting failed
                if result != 0 {
                    anyhow::bail!("Failed to post bboxes")
                }
            }

            Ok(())
        }).await
            .context("Error trying to post bboxes")?
            .context("Error posting bboxes")?;

        Ok(())
    }

    // Callbacks
    extern "C" fn _source_frames_callback(
        source_id: c_int,
        frame: *const u8,
        width: c_int,
        height: c_int,
        pts: c_ulonglong,
    ) {
        let source_id = source_id.to_string();
        let width = width as u32;
        let height = height as u32;
        let frame_size = (width * height * 3) as usize;

        // We spawn a task in our threadpool to not block the C callback
        
        if let Ok(rgb_frame) = ClientVideo::get_c_array(frame, frame_size) {
            if let Ok(runtime) = crate::get_tokio_runtime() {
                runtime.spawn(async move {
                    match source::get_source_processor(&source_id).await {
                        Err(e) => {
                            tracing::error!(
                                error=e.to_string(),
                                source_id=source_id, 
                                "Source processor is not available"
                            )
                        },
                        Ok(processor) => {
                            processor.process_frame(rgb_frame, height, width, pts).await;
                        }
                    }
                });
            } else {
                tracing::error!(
                    source_id=source_id, 
                    "Cannot process frame on application runtime"
                );
            }
        } else {
            tracing::error!(
                source_id=source_id, 
                "RGB Frame is invalid"
            );
            return;
        }
        

    }
    extern "C" fn _source_stopped_callback(source_id: c_int) {
        tracing::info!(
            source_id=source_id, 
            "Source stopped!"
        );
    }

    extern "C" fn _source_name_callback(source_id: c_int, source_name: *const c_char) {
        let source_name = ClientVideo::get_c_string(source_name)
            .unwrap_or("UNKNOWN".to_string());

        tracing::info!(
            source_id=source_id, 
            source_name=source_name, 
            "Got source name"
        );
    }

    extern "C" fn _source_status_callback(source_id: c_int, source_status: c_int) {
        let source_status = match source_status {
            0 => "OK - Stream is active",
            1 => "ERROR - Not streaming",
            2 => "ERROR - Source not found",
            3 => "ERROR - Connection error",
            4 => "ERROR - Decode error",
            _ => "UNKNOWN status",
        };

        tracing::info!(
            source_id=source_id, 
            source_status=source_status, 
            "Got source status"
        );
    }

    // Helper functions
    fn free_c_ptr<T>(ptr: *const T) -> Result<()> {
        let client_video = get_client_video()?;

        unsafe {
            let lib_free_c_ptr: Symbol<FreeCPtrFn> = client_video.library()
                .get(b"FreeCPtr")
                .context("Cannot get 'FreeCPtr' function")?;

            // Call library function
            lib_free_c_ptr(ptr as *const c_void);
        }

        Ok(())
    }

    fn get_c_string(ptr: *const c_char) -> Result<String> {
        if ptr.is_null() {
            anyhow::bail!("C string is invalid!")
        }

        let string = unsafe {
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        };

        // Try freeing the pointer
        if let Err(e) = ClientVideo::free_c_ptr(ptr) {
            tracing::info!(
                error=e.to_string(),
                "Error freeing c string pointer"
            )
        }

        Ok(string)
    }

    fn get_c_array<T: Clone>(ptr: *const T, len: usize) -> Result<Vec<T>> {
        if ptr.is_null() {
            anyhow::bail!("C list is invalid!")
        }
        
        // Create a slice from the raw pointer
        let slice = unsafe {
            slice::from_raw_parts(ptr, len)
        };
        
        Ok(slice.to_vec())
    }
}

impl ClientVideo {
    fn library(&self) -> &Library {
        &self.library
    }
}