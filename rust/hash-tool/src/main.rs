//! hash-tool — perceptual image hashing for the czkawka duplicate-compare page.
//!
//! Decodes each input image straight to grayscale (JPEGs via zune-jpeg's Luma
//! output, which skips chroma decoding entirely), pre-shrinks it once with the
//! SIMD `fast_image_resize` crate (the same trick czkawka uses for its big
//! similar-images speedup), then hashes the original plus its horizontal flop
//! (mirror) so the comparer can match mirrored duplicates. Upside-down
//! duplicates are deliberately not covered — they don't occur in practice and
//! a vertical-flip hash would cost a third more work. Uses the same
//! `image_hasher` crate as czkawka itself, so every hash algorithm / resize
//! filter / hash size the UI offers behaves like czkawka's. The pre-shrink
//! means hashes are not bit-identical to a full-resolution single-pass hash,
//! but the comparer's Hamming thresholds absorb the difference; bump the cache
//! version in src/fs/paths.ts if this pipeline changes again.
//!
//! Input (--jobs): { "hashAlg", "imageFilter", "hashSize",
//!                   "images": [{ "key", "path" }] }
//! Output (stdout): { "results": [{ key, hash, flopHash, width, height }],
//!                    "errors": [{ key, path, error }] }
//! Progress: `progress:`-prefixed lines on stderr (forwarded by the TS spawner).

use std::sync::atomic::{AtomicUsize, Ordering};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use clap::Parser;
use fast_image_resize as fir;
use image::{ImageBuffer, Luma, imageops};
use image_hasher::{FilterType, HashAlg, HasherConfig};
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(about = "Perceptual image hashing (original + flip + flop) via image_hasher")]
struct Args {
    /// Path to the jobs JSON file.
    #[arg(long)]
    jobs: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct JobsInput {
    hash_alg: String,
    image_filter: String,
    hash_size: u32,
    images: Vec<ImageJob>,
}

#[derive(Deserialize)]
struct ImageJob {
    /// Opaque cache key (content hash) echoed back with the result.
    key: String,
    path: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HashResult {
    key: String,
    hash: String,
    flop_hash: String,
    width: u32,
    height: u32,
}

#[derive(Serialize)]
struct HashError {
    key: String,
    path: String,
    error: String,
}

#[derive(Serialize)]
struct Output {
    results: Vec<HashResult>,
    errors: Vec<HashError>,
}

fn parse_alg(s: &str) -> HashAlg {
    match s {
        "Mean" => HashAlg::Mean,
        "Gradient" => HashAlg::Gradient,
        "VertGradient" => HashAlg::VertGradient,
        "DoubleGradient" => HashAlg::DoubleGradient,
        "Blockhash" => HashAlg::Blockhash,
        "Median" => HashAlg::Median,
        other => panic!("unknown hash algorithm: {other}"),
    }
}

fn parse_filter(s: &str) -> FilterType {
    match s {
        "Lanczos3" => FilterType::Lanczos3,
        "Nearest" => FilterType::Nearest,
        "Triangle" => FilterType::Triangle,
        "Gaussian" => FilterType::Gaussian,
        "CatmullRom" | "Catmullrom" => FilterType::CatmullRom,
        other => panic!("unknown resize filter: {other}"),
    }
}

/// fast_image_resize equivalent of the UI's resize-filter choice, used for the
/// SIMD pre-shrink (Triangle is fir's Bilinear).
fn parse_fir_alg(s: &str) -> fir::ResizeAlg {
    use fir::{FilterType as F, ResizeAlg as A};
    match s {
        "Lanczos3" => A::Convolution(F::Lanczos3),
        "Nearest" => A::Nearest,
        "Triangle" => A::Convolution(F::Bilinear),
        "Gaussian" => A::Convolution(F::Gaussian),
        "CatmullRom" | "Catmullrom" => A::Convolution(F::CatmullRom),
        other => panic!("unknown resize filter: {other}"),
    }
}

fn main() {
    let args = Args::parse();
    let input_text = std::fs::read_to_string(&args.jobs)
        .unwrap_or_else(|e| panic!("read jobs file {}: {}", args.jobs, e));
    let input: JobsInput = serde_json::from_str(&input_text).expect("parse jobs JSON");

    let alg = parse_alg(&input.hash_alg);
    let filter = parse_filter(&input.image_filter);
    let fir_alg = parse_fir_alg(&input.image_filter);
    let size = input.hash_size;

    let total = input.images.len();
    let step = (total / 50).max(1);
    eprintln!("progress: Hashing {total} image(s) ({}, size {size})...", input.hash_alg);

    let done = AtomicUsize::new(0);
    let outcomes: Vec<Result<HashResult, HashError>> = input
        .images
        .par_iter()
        .map(|job| {
            let outcome = hash_one(job, alg, filter, fir_alg, size);
            let c = done.fetch_add(1, Ordering::Relaxed) + 1;
            if c % step == 0 || c == total {
                eprintln!("progress: Hashed {c}/{total} images...");
            }
            outcome
        })
        .collect();

    let mut out = Output { results: Vec::with_capacity(total), errors: Vec::new() };
    for o in outcomes {
        match o {
            Ok(r) => out.results.push(r),
            Err(e) => out.errors.push(e),
        }
    }

    serde_json::to_writer(std::io::stdout().lock(), &out).expect("write JSON");
}

/// Pre-shrink target edge: comfortably above every hash size's internal
/// resample resolution, small enough that the two per-orientation hashes
/// (each a grayscale resample of this buffer) cost nothing.
fn preshrink_edge(hash_size: u32) -> u32 {
    (hash_size * 8).max(64)
}

fn hash_one(
    job: &ImageJob,
    alg: HashAlg,
    filter: FilterType,
    fir_alg: fir::ResizeAlg,
    size: u32,
) -> Result<HashResult, HashError> {
    let err = |error: String| HashError {
        key: job.key.clone(),
        path: job.path.clone(),
        error,
    };
    let (gray, width, height) = decode_luma(&job.path).map_err(err)?;

    // One grayscale decode + one SIMD downscale; the flop variant then
    // operates on the small buffer instead of re-processing the full image.
    let edge = preshrink_edge(size);
    let small: ImageBuffer<Luma<u8>, Vec<u8>> = if gray.width() > edge || gray.height() > edge {
        let (tw, th) = (gray.width().min(edge), gray.height().min(edge));
        let (gw, gh) = (gray.width(), gray.height());
        let src = fir::images::Image::from_vec_u8(gw, gh, gray.into_raw(), fir::PixelType::U8)
            .map_err(|e| err(e.to_string()))?;
        let mut dst = fir::images::Image::new(tw, th, fir::PixelType::U8);
        fir::Resizer::new()
            .resize(&src, &mut dst, &fir::ResizeOptions::new().resize_alg(fir_alg))
            .map_err(|e| err(e.to_string()))?;
        ImageBuffer::from_raw(tw, th, dst.into_vec()).expect("buffer size matches dimensions")
    } else {
        gray
    };

    // Hasher construction is cheap (no DCT preprocessing); build per call so
    // the parallel map needs no shared state.
    let hasher = HasherConfig::new()
        .hash_size(size, size)
        .hash_alg(alg)
        .resize_filter(filter)
        .to_hasher();

    let hash = hasher.hash_image(&small);
    let flop_hash = hasher.hash_image(&imageops::flip_horizontal(&small));

    Ok(HashResult {
        key: job.key.clone(),
        hash: B64.encode(hash.as_bytes()),
        flop_hash: B64.encode(flop_hash.as_bytes()),
        width,
        height,
    })
}

/// Decode to grayscale. JPEGs go through zune-jpeg asking for Luma output
/// directly — the Y channel IS Rec.601 luma, so this skips the chroma IDCT,
/// chroma upsampling, YCbCr→RGB conversion, and the RGB→gray pass (and the
/// full-size RGB allocation). Anything else — or any zune failure (CMYK
/// jpegs, truncated files) — falls back to the general image crate path.
fn decode_luma(path: &str) -> Result<(ImageBuffer<Luma<u8>, Vec<u8>>, u32, u32), String> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        if let Ok(data) = std::fs::read(path) {
            let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::Luma);
            let mut dec = JpegDecoder::new_with_options(&data, opts);
            if let Ok(pixels) = dec.decode() {
                if let Some((w, h)) = dec.dimensions() {
                    let (w, h) = (w as u32, h as u32);
                    if let Some(buf) = ImageBuffer::from_raw(w, h, pixels) {
                        return Ok((buf, w, h));
                    }
                }
            }
        }
    }
    let img = image::open(path).map_err(|e| e.to_string())?;
    let (w, h) = (img.width(), img.height());
    Ok((img.into_luma8(), w, h))
}
