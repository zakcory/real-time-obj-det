use anyhow::{Context, Result};
use image::{GenericImageView, ImageReader};

// Custom modules
pub mod config;
pub mod queue;
pub mod elastic;

/// used to get image from path, returns as raw bytes
pub fn get_image_raw(path: &str) -> Result<(Vec<u8>, u32, u32)> {
    let image = ImageReader::open(path)
        .context("Error opening image from path")?
        .decode()
        .context("Error decoding image")?;

    // Get dimensions
    let (width, height) = image.dimensions();

    // Convert to RGB8 if needed
    let img_rgb8 = image.to_rgb8();

    // Get raw pixel data
    Ok((img_rgb8.into_raw(), height, width))
}