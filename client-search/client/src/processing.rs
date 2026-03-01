//! Responsible for pre/post processing images before inference.
//! Performs operations on raw frames/inference results with SIMD optimizations

use anyhow::Result;
use std::sync::OnceLock;
use tokio::time::Instant;
use serde::Serialize;

// Custom modules
pub mod dino;
pub mod search;
use crate::utils::config::InferencePrecision;

/// Normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const PAD_GRAY_COLOR: usize = 114;

/// Represents raw frame before performing inference on it
#[derive(Clone, Debug)]
pub struct RawFrame {
    pub data: Vec<u8>,
    pub height: u32,
    pub width: u32,
    pub added: Instant
}

/// Represents embedding output from the model inference results
#[derive(Clone, Serialize)]
pub struct ResultEmbedding {
    pub data: Vec<f32>
}

impl ResultEmbedding {
    pub fn get_raw_bytes(&self) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>()
            )
        }.to_vec()
    }
}

/// Lookup table for converting values from FP16 to FP32
pub static F16_TO_F32_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
/// Lookup table for F32 to F16 conversion
pub static F32_TO_F16_LUT: OnceLock<Box<[u16; 32768]>> = OnceLock::new();
/// Lookup table for converting pixel values to FP16
pub static F16_LUT: OnceLock<Box<[u16; 256]>> = OnceLock::new();
/// Lookup table for converting pixel values to FP32
pub static F32_LUT: OnceLock<Box<[f32; 256]>> = OnceLock::new();

/// Lookup tables for ImageNet normalization (u8 -> normalized f32)
pub static IMAGENET_R_F32_LUT: OnceLock<Box<[f32; 256]>> = OnceLock::new();
pub static IMAGENET_G_F32_LUT: OnceLock<Box<[f32; 256]>> = OnceLock::new();
pub static IMAGENET_B_F32_LUT: OnceLock<Box<[f32; 256]>> = OnceLock::new();

/// Lookup tables for ImageNet normalization (u8 -> normalized f16)
pub static IMAGENET_R_F16_LUT: OnceLock<Box<[u16; 256]>> = OnceLock::new();
pub static IMAGENET_G_F16_LUT: OnceLock<Box<[u16; 256]>> = OnceLock::new();
pub static IMAGENET_B_F16_LUT: OnceLock<Box<[u16; 256]>> = OnceLock::new();

/// Create static lookup table for high speed conversion
fn create_f16_to_f32_lut() -> Box<[f32; 65536]> {
    let mut lut = Box::new([0.0f32; 65536]);
        
    for i in 0u16..=65535 {
        let sign = (i >> 15) & 0x1;
        let exp = (i >> 10) & 0x1f;
        let frac = i & 0x3ff;
        
        lut[i as usize] = if exp == 0 {
            if frac == 0 {
                if sign == 1 { -0.0 } else { 0.0 }
            } else {
                // Denormal
                let mut val = frac as f32 / 1024.0 / 16384.0;
                if sign == 1 { val = -val; }
                val
            }
        } else if exp == 31 {
            // Infinity or NaN
            if frac == 0 {
                if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            // Normal numbers
            let exp_f32 = (exp as i32 - 15 + 127) as u32;
            let frac_f32 = (frac as u32) << 13;
            let bits = (sign as u32) << 31 | exp_f32 << 23 | frac_f32;
            f32::from_bits(bits)
        };
    }
    
    lut
}

pub fn get_f16_to_f32_lut(val: u16) -> f32 {
    F16_TO_F32_LUT
        .get_or_init(create_f16_to_f32_lut)[val as usize]
}

/// Create static lookup table for F32 to F16 conversion
fn create_f32_to_f16_lut() -> Box<[u16; 32768]> {
    let mut lut = Box::new([0u16; 32768]);
    
    const MIN_VAL: f32 = -4.0;
    const MAX_VAL: f32 = 4.0;
    const RANGE: f32 = MAX_VAL - MIN_VAL;
    const STEP: f32 = RANGE / 32768.0;
    
    for i in 0..32768 {
        let val = MIN_VAL + (i as f32) * STEP;
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7fffff;
        
        lut[i] = if exp == 0 {
            sign as u16
        } else {
            let exp_adj = exp - 127 + 15;
            if exp_adj >= 31 {
                (sign | 0x7c00) as u16
            } else if exp_adj <= 0 {
                sign as u16
            } else {
                let mantissa_adj = mantissa >> 13;
                (sign | ((exp_adj as u32) << 10) | mantissa_adj) as u16
            }
        };
    }
    
    lut
}

fn get_f32_to_f16_lut(val: f32) -> u16 {
    const MIN_VAL: f32 = -4.0;
    const MAX_VAL: f32 = 4.0;
    const RANGE: f32 = MAX_VAL - MIN_VAL;
    
    let clamped_val = val.clamp(MIN_VAL, MAX_VAL);
    let index = ((clamped_val - MIN_VAL) / RANGE * 32767.0) as usize;
    let index = index.min(32767);
    
    F32_TO_F16_LUT
        .get_or_init(create_f32_to_f16_lut)[index]
}

/// Create static lookup table for high speed conversion
fn create_f16_lut() -> Box<[u16; 256]> {
    let mut lut = Box::new([0u16; 256]);
    for i in 0..256 {
        let normalized = i as f32 / 255.0;
        let bits = normalized.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7fffff;
        lut[i] = if exp == 0 {
            sign as u16
        } else {
            let exp_adj = exp - 127 + 15;
            if exp_adj >= 31 {
                (sign | 0x7c00) as u16
            } else if exp_adj <= 0 {
                sign as u16
            } else {
                let mantissa_adj = mantissa >> 13;
                (sign | ((exp_adj as u32) << 10) | mantissa_adj) as u16
            }
        };
    }
    lut
}

pub fn get_f16_lut() -> &'static [u16; 256] {
    F16_LUT
        .get_or_init(create_f16_lut)
}

/// Create static lookup table for high speed conversion
fn create_f32_lut() -> Box<[f32; 256]> {
    let mut lut = Box::new([0.0f32; 256]);
    for i in 0..256 {
        lut[i] = i as f32 / 255.0;
    }
    lut
}

pub fn get_f32_lut() -> &'static [f32; 256] {
    F32_LUT
        .get_or_init(create_f32_lut)
}

/// Create ImageNet normalization lookup tables for R channel (F32)
fn create_imagenet_r_f32_lut() -> Box<[f32; 256]> {
    let mut lut = Box::new([0.0f32; 256]);
    let r_std_inv = 1.0 / IMAGENET_STD[0];
    for i in 0..256 {
        let normalized = (i as f32 / 255.0 - IMAGENET_MEAN[0]) * r_std_inv;
        lut[i] = normalized;
    }
    lut
}

/// Create ImageNet normalization lookup tables for G channel (F32)
fn create_imagenet_g_f32_lut() -> Box<[f32; 256]> {
    let mut lut = Box::new([0.0f32; 256]);
    let g_std_inv = 1.0 / IMAGENET_STD[1];
    for i in 0..256 {
        let normalized = (i as f32 / 255.0 - IMAGENET_MEAN[1]) * g_std_inv;
        lut[i] = normalized;
    }
    lut
}

/// Create ImageNet normalization lookup tables for B channel (F32)
fn create_imagenet_b_f32_lut() -> Box<[f32; 256]> {
    let mut lut = Box::new([0.0f32; 256]);
    let b_std_inv = 1.0 / IMAGENET_STD[2];
    for i in 0..256 {
        let normalized = (i as f32 / 255.0 - IMAGENET_MEAN[2]) * b_std_inv;
        lut[i] = normalized;
    }
    lut
}

/// Create ImageNet normalization lookup tables for R channel (F16)
fn create_imagenet_r_f16_lut() -> Box<[u16; 256]> {
    let mut lut = Box::new([0u16; 256]);
    let r_std_inv = 1.0 / IMAGENET_STD[0];
    for i in 0..256 {
        let normalized = (i as f32 / 255.0 - IMAGENET_MEAN[0]) * r_std_inv;
        lut[i] = get_f32_to_f16_lut(normalized);
    }
    lut
}

/// Create ImageNet normalization lookup tables for G channel (F16)
fn create_imagenet_g_f16_lut() -> Box<[u16; 256]> {
    let mut lut = Box::new([0u16; 256]);
    let g_std_inv = 1.0 / IMAGENET_STD[1];
    for i in 0..256 {
        let normalized = (i as f32 / 255.0 - IMAGENET_MEAN[1]) * g_std_inv;
        lut[i] = get_f32_to_f16_lut(normalized);
    }
    lut
}

/// Create ImageNet normalization lookup tables for B channel (F16)
fn create_imagenet_b_f16_lut() -> Box<[u16; 256]> {
    let mut lut = Box::new([0u16; 256]);
    let b_std_inv = 1.0 / IMAGENET_STD[2];
    for i in 0..256 {
        let normalized = (i as f32 / 255.0 - IMAGENET_MEAN[2]) * b_std_inv;
        lut[i] = get_f32_to_f16_lut(normalized);
    }
    lut
}

/// Get ImageNet-normalized lookup tables
pub fn get_imagenet_r_f32_lut() -> &'static [f32; 256] {
    IMAGENET_R_F32_LUT.get_or_init(create_imagenet_r_f32_lut)
}

pub fn get_imagenet_g_f32_lut() -> &'static [f32; 256] {
    IMAGENET_G_F32_LUT.get_or_init(create_imagenet_g_f32_lut)
}

pub fn get_imagenet_b_f32_lut() -> &'static [f32; 256] {
    IMAGENET_B_F32_LUT.get_or_init(create_imagenet_b_f32_lut)
}

pub fn get_imagenet_r_f16_lut() -> &'static [u16; 256] {
    IMAGENET_R_F16_LUT.get_or_init(create_imagenet_r_f16_lut)
}

pub fn get_imagenet_g_f16_lut() -> &'static [u16; 256] {
    IMAGENET_G_F16_LUT.get_or_init(create_imagenet_g_f16_lut)
}

pub fn get_imagenet_b_f16_lut() -> &'static [u16; 256] {
    IMAGENET_B_F16_LUT.get_or_init(create_imagenet_b_f16_lut)
}

#[derive(Debug, Clone, Copy)]
pub struct LetterboxParams {
    pub new_width: u32,
    pub new_height: u32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub scale: f32,
    pub inv_scale: f32,
}

/// Calculate letterbox parameters for image resizing
pub fn calculate_letterbox(in_h: u32, in_w: u32, target_size: u32) -> LetterboxParams {
    let scale = (target_size as f32) / (in_h.max(in_w) as f32);
    let new_width = (in_w as f32 * scale) as u32;
    let new_height = (in_h as f32 * scale) as u32;
    let pad_x = (target_size - new_width) / 2;
    let pad_y = (target_size - new_height) / 2;

    LetterboxParams {
        new_width,
        new_height,
        pad_x,
        pad_y,
        scale,
        inv_scale: 1.0 / scale,
    }
}

///
/// Performs a single-pass, fused nearest-neighbor resize, letterbox,
/// pixel normalization (x / 255.0) and ImageNet normalization.
///
/// * `input`: Raw `u8` RGB interleaved pixel data.
/// * `in_h`, `in_w`: Dimensions of the `input` image.
/// * `target_h`, `target_w`: Dimensions of the `output` buffer.
/// * `precision`: The desired output precision (FP32 or FP16).
///
/// Returns a new `Vec<u8>` containing the final FP32 or FP16 planar data.
///
pub fn resize_letterbox_and_normalize_imagenet(
    input: &[u8],
    in_h: u32,
    in_w: u32,
    target_h: u32,
    target_w: u32,
    precision: InferencePrecision,
) -> Result<Vec<u8>> {
    // 1. Calculate letterbox params
    let letterbox = calculate_letterbox(in_h, in_w, target_h.max(target_w));
    let num_pixels = (target_h * target_w) as usize;

    // 2. Allocate the *FINAL* output buffer ONCE
    let mut output: Vec<u8> = match precision {
        InferencePrecision::FP16 => vec![0u8; num_pixels * 3 * 2],
        InferencePrecision::FP32 => vec![0u8; num_pixels * 3 * 4],
    };

    // 3. Pre-calculate x-offsets for the source image
    let mut x_offsets: Vec<u32> = Vec::with_capacity(letterbox.new_width as usize);
    for x in 0..letterbox.new_width {
        x_offsets.push(((x as f32 * letterbox.inv_scale) as u32).min(in_w - 1) * 3);
    }

    let in_ptr = input.as_ptr();

    // 4. Perform fused resize, normalization (pixel + ImageNet), and planar conversion
    match precision {
        InferencePrecision::FP16 => {
            // Get pre-computed ImageNet normalization LUTs
            let r_lut = get_imagenet_r_f16_lut();
            let g_lut = get_imagenet_g_f16_lut();
            let b_lut = get_imagenet_b_f16_lut();
            
            let pad_val_r_f16 = r_lut[PAD_GRAY_COLOR];
            let pad_val_g_f16 = g_lut[PAD_GRAY_COLOR];
            let pad_val_b_f16 = b_lut[PAD_GRAY_COLOR];
            
            let out_ptr = output.as_mut_ptr() as *mut u16;
            let (out_r, out_g, out_b) = unsafe {
                (
                    std::slice::from_raw_parts_mut(out_ptr, num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels), num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels * 2), num_pixels),
                )
            };

            // Pre-fill with normalized padding color
            out_r.fill(pad_val_r_f16);
            out_g.fill(pad_val_g_f16);
            out_b.fill(pad_val_b_f16);

            // Hoist constants
            let pad_x = letterbox.pad_x;
            let pad_y = letterbox.pad_y;
            let inv_scale = letterbox.inv_scale;
            let in_w_3 = in_w * 3;
            
            // Get raw pointers for unchecked access
            let r_lut_ptr = r_lut.as_ptr();
            let g_lut_ptr = g_lut.as_ptr();
            let b_lut_ptr = b_lut.as_ptr();
            let x_offsets_ptr = x_offsets.as_ptr();

            // Write real pixels with ImageNet normalization using LUTs
            for y in 0..letterbox.new_height {
                let src_y = ((y as f32 * inv_scale) as u32).min(in_h - 1);
                let src_row_offset = src_y * in_w_3;
                let dst_row_base = (y + pad_y) * target_w + pad_x;
                
                let width = letterbox.new_width;
                let mut x = 0;
                
                // Process 4 pixels at a time (loop unrolling)
                while x + 4 <= width {
                    unsafe {
                        let x0_offset = *x_offsets_ptr.add(x as usize);
                        let x1_offset = *x_offsets_ptr.add((x + 1) as usize);
                        let x2_offset = *x_offsets_ptr.add((x + 2) as usize);
                        let x3_offset = *x_offsets_ptr.add((x + 3) as usize);
                        
                        let src_idx0 = (src_row_offset + x0_offset) as usize;
                        let src_idx1 = (src_row_offset + x1_offset) as usize;
                        let src_idx2 = (src_row_offset + x2_offset) as usize;
                        let src_idx3 = (src_row_offset + x3_offset) as usize;
                        
                        let dst_idx0 = (dst_row_base + x) as usize;
                        let dst_idx1 = (dst_row_base + x + 1) as usize;
                        let dst_idx2 = (dst_row_base + x + 2) as usize;
                        let dst_idx3 = (dst_row_base + x + 3) as usize;
                        
                        // Pixel 0
                        *out_r.get_unchecked_mut(dst_idx0) = *r_lut_ptr.add(*in_ptr.add(src_idx0) as usize);
                        *out_g.get_unchecked_mut(dst_idx0) = *g_lut_ptr.add(*in_ptr.add(src_idx0 + 1) as usize);
                        *out_b.get_unchecked_mut(dst_idx0) = *b_lut_ptr.add(*in_ptr.add(src_idx0 + 2) as usize);
                        
                        // Pixel 1
                        *out_r.get_unchecked_mut(dst_idx1) = *r_lut_ptr.add(*in_ptr.add(src_idx1) as usize);
                        *out_g.get_unchecked_mut(dst_idx1) = *g_lut_ptr.add(*in_ptr.add(src_idx1 + 1) as usize);
                        *out_b.get_unchecked_mut(dst_idx1) = *b_lut_ptr.add(*in_ptr.add(src_idx1 + 2) as usize);
                        
                        // Pixel 2
                        *out_r.get_unchecked_mut(dst_idx2) = *r_lut_ptr.add(*in_ptr.add(src_idx2) as usize);
                        *out_g.get_unchecked_mut(dst_idx2) = *g_lut_ptr.add(*in_ptr.add(src_idx2 + 1) as usize);
                        *out_b.get_unchecked_mut(dst_idx2) = *b_lut_ptr.add(*in_ptr.add(src_idx2 + 2) as usize);
                        
                        // Pixel 3
                        *out_r.get_unchecked_mut(dst_idx3) = *r_lut_ptr.add(*in_ptr.add(src_idx3) as usize);
                        *out_g.get_unchecked_mut(dst_idx3) = *g_lut_ptr.add(*in_ptr.add(src_idx3 + 1) as usize);
                        *out_b.get_unchecked_mut(dst_idx3) = *b_lut_ptr.add(*in_ptr.add(src_idx3 + 2) as usize);
                    }
                    x += 4;
                }
                
                // Handle remaining pixels
                while x < width {
                    unsafe {
                        let x_offset = *x_offsets_ptr.add(x as usize);
                        let src_idx = (src_row_offset + x_offset) as usize;
                        let dst_idx = (dst_row_base + x) as usize;
                        
                        *out_r.get_unchecked_mut(dst_idx) = *r_lut_ptr.add(*in_ptr.add(src_idx) as usize);
                        *out_g.get_unchecked_mut(dst_idx) = *g_lut_ptr.add(*in_ptr.add(src_idx + 1) as usize);
                        *out_b.get_unchecked_mut(dst_idx) = *b_lut_ptr.add(*in_ptr.add(src_idx + 2) as usize);
                    }
                    x += 1;
                }
            }
        }
        InferencePrecision::FP32 => {
            // Get pre-computed ImageNet normalization LUTs
            let r_lut = get_imagenet_r_f32_lut();
            let g_lut = get_imagenet_g_f32_lut();
            let b_lut = get_imagenet_b_f32_lut();
            
            let pad_val_r = r_lut[PAD_GRAY_COLOR];
            let pad_val_g = g_lut[PAD_GRAY_COLOR];
            let pad_val_b = b_lut[PAD_GRAY_COLOR];
            
            let out_ptr = output.as_mut_ptr() as *mut f32;
            let (out_r, out_g, out_b) = unsafe {
                (
                    std::slice::from_raw_parts_mut(out_ptr, num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels), num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels * 2), num_pixels),
                )
            };

            // Pre-fill with normalized padding color
            out_r.fill(pad_val_r);
            out_g.fill(pad_val_g);
            out_b.fill(pad_val_b);

            // Hoist constants
            let pad_x = letterbox.pad_x;
            let pad_y = letterbox.pad_y;
            let inv_scale = letterbox.inv_scale;
            let in_w_3 = in_w * 3;
            
            // Get raw pointers for unchecked access
            let r_lut_ptr = r_lut.as_ptr();
            let g_lut_ptr = g_lut.as_ptr();
            let b_lut_ptr = b_lut.as_ptr();
            let x_offsets_ptr = x_offsets.as_ptr();

            // Write real pixels with ImageNet normalization using LUTs
            for y in 0..letterbox.new_height {
                let src_y = ((y as f32 * inv_scale) as u32).min(in_h - 1);
                let src_row_offset = src_y * in_w_3;
                let dst_row_base = (y + pad_y) * target_w + pad_x;
                
                let width = letterbox.new_width;
                let mut x = 0;
                
                // Process 4 pixels at a time (loop unrolling)
                while x + 4 <= width {
                    unsafe {
                        let x0_offset = *x_offsets_ptr.add(x as usize);
                        let x1_offset = *x_offsets_ptr.add((x + 1) as usize);
                        let x2_offset = *x_offsets_ptr.add((x + 2) as usize);
                        let x3_offset = *x_offsets_ptr.add((x + 3) as usize);
                        
                        let src_idx0 = (src_row_offset + x0_offset) as usize;
                        let src_idx1 = (src_row_offset + x1_offset) as usize;
                        let src_idx2 = (src_row_offset + x2_offset) as usize;
                        let src_idx3 = (src_row_offset + x3_offset) as usize;
                        
                        let dst_idx0 = (dst_row_base + x) as usize;
                        let dst_idx1 = (dst_row_base + x + 1) as usize;
                        let dst_idx2 = (dst_row_base + x + 2) as usize;
                        let dst_idx3 = (dst_row_base + x + 3) as usize;
                        
                        // Pixel 0
                        *out_r.get_unchecked_mut(dst_idx0) = *r_lut_ptr.add(*in_ptr.add(src_idx0) as usize);
                        *out_g.get_unchecked_mut(dst_idx0) = *g_lut_ptr.add(*in_ptr.add(src_idx0 + 1) as usize);
                        *out_b.get_unchecked_mut(dst_idx0) = *b_lut_ptr.add(*in_ptr.add(src_idx0 + 2) as usize);
                        
                        // Pixel 1
                        *out_r.get_unchecked_mut(dst_idx1) = *r_lut_ptr.add(*in_ptr.add(src_idx1) as usize);
                        *out_g.get_unchecked_mut(dst_idx1) = *g_lut_ptr.add(*in_ptr.add(src_idx1 + 1) as usize);
                        *out_b.get_unchecked_mut(dst_idx1) = *b_lut_ptr.add(*in_ptr.add(src_idx1 + 2) as usize);
                        
                        // Pixel 2
                        *out_r.get_unchecked_mut(dst_idx2) = *r_lut_ptr.add(*in_ptr.add(src_idx2) as usize);
                        *out_g.get_unchecked_mut(dst_idx2) = *g_lut_ptr.add(*in_ptr.add(src_idx2 + 1) as usize);
                        *out_b.get_unchecked_mut(dst_idx2) = *b_lut_ptr.add(*in_ptr.add(src_idx2 + 2) as usize);
                        
                        // Pixel 3
                        *out_r.get_unchecked_mut(dst_idx3) = *r_lut_ptr.add(*in_ptr.add(src_idx3) as usize);
                        *out_g.get_unchecked_mut(dst_idx3) = *g_lut_ptr.add(*in_ptr.add(src_idx3 + 1) as usize);
                        *out_b.get_unchecked_mut(dst_idx3) = *b_lut_ptr.add(*in_ptr.add(src_idx3 + 2) as usize);
                    }
                    x += 4;
                }
                
                // Handle remaining pixels
                while x < width {
                    unsafe {
                        let x_offset = *x_offsets_ptr.add(x as usize);
                        let src_idx = (src_row_offset + x_offset) as usize;
                        let dst_idx = (dst_row_base + x) as usize;
                        
                        *out_r.get_unchecked_mut(dst_idx) = *r_lut_ptr.add(*in_ptr.add(src_idx) as usize);
                        *out_g.get_unchecked_mut(dst_idx) = *g_lut_ptr.add(*in_ptr.add(src_idx + 1) as usize);
                        *out_b.get_unchecked_mut(dst_idx) = *b_lut_ptr.add(*in_ptr.add(src_idx + 2) as usize);
                    }
                    x += 1;
                }
            }
        }
    }

    Ok(output)
}