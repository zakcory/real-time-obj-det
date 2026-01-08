use anyhow::{Result, Context};
use image::{GenericImageView, ImageFormat};
use nvml_wrapper::Nvml;

// Custom modules
pub mod config;
pub mod elastic;

/// Parses image bytes to extract an image
pub fn parse_image(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let format = detect_format(data)?;
    
    let img = image::load_from_memory_with_format(data, format)
        .context("Failed to decode image")?;
    
    let (width, height) = img.dimensions();
    Ok((img.into_rgb8().into_raw(), width, height))
}

fn detect_format(data: &[u8]) -> Result<ImageFormat> {
    if data.len() < 8 {
        anyhow::bail!("Image data too small");
    }
    
    // JPEG: FF D8 FF
    if data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
        return Ok(ImageFormat::Jpeg);
    }
    
    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if &data[0..8] == b"\x89PNG\r\n\x1a\n" {
        return Ok(ImageFormat::Png);
    }
    
    // WebP: RIFF....WEBP
    if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        return Ok(ImageFormat::WebP);
    }
    
    // GIF: GIF87a or GIF89a
    if &data[0..6] == b"GIF87a" || &data[0..6] == b"GIF89a" {
        return Ok(ImageFormat::Gif);
    }
    
    // Fallback to auto-detection
    image::guess_format(data).context("Unknown image format")
}
/// Get GPU name
pub fn get_gpu_name() -> Result<String> {
    let nvml = Nvml::init()
        .context("Error initiating NVML wrapper")?;
    let gpu = nvml.device_by_index(0)
        .context("Error getting GPU")?;
    let gpu_name = gpu.name()
        .context("Error getting GPU name")?;
    Ok(gpu_name)
}
    