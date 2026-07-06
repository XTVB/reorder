//! hash-tool — perceptual image hashing for the czkawka duplicate-compare page.
//!
//! Decodes each input image once, then hashes the original plus its vertical
//! flip and horizontal flop (mirror) so the comparer can match reflected
//! duplicates. Uses the same `image_hasher` crate as czkawka itself, so every
//! hash algorithm / resize filter / hash size the UI offers matches czkawka's
//! output exactly.
//!
//! Input (--jobs): { "hashAlg", "imageFilter", "hashSize",
//!                   "images": [{ "key", "path" }] }
//! Output (stdout): { "results": [{ key, hash, flipHash, flopHash,
//!                    width, height }], "errors": [{ key, path, error }] }
//! Progress: `progress:`-prefixed lines on stderr (forwarded by the TS spawner).

use std::sync::atomic::{AtomicUsize, Ordering};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use clap::Parser;
use image_hasher::{FilterType, HashAlg, HasherConfig};
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
    flip_hash: String,
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

fn main() {
    let args = Args::parse();
    let input_text = std::fs::read_to_string(&args.jobs)
        .unwrap_or_else(|e| panic!("read jobs file {}: {}", args.jobs, e));
    let input: JobsInput = serde_json::from_str(&input_text).expect("parse jobs JSON");

    let alg = parse_alg(&input.hash_alg);
    let filter = parse_filter(&input.image_filter);
    let size = input.hash_size;

    let total = input.images.len();
    let step = (total / 50).max(1);
    eprintln!("progress: Hashing {total} image(s) ({}, size {size})...", input.hash_alg);

    let done = AtomicUsize::new(0);
    let outcomes: Vec<Result<HashResult, HashError>> = input
        .images
        .par_iter()
        .map(|job| {
            let outcome = hash_one(job, alg, filter, size);
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

fn hash_one(
    job: &ImageJob,
    alg: HashAlg,
    filter: FilterType,
    size: u32,
) -> Result<HashResult, HashError> {
    let img = image::open(&job.path).map_err(|e| HashError {
        key: job.key.clone(),
        path: job.path.clone(),
        error: e.to_string(),
    })?;
    let (width, height) = (img.width(), img.height());

    // Hasher construction is cheap (no DCT preprocessing); build per call so
    // the parallel map needs no shared state.
    let hasher = HasherConfig::new()
        .hash_size(size, size)
        .hash_alg(alg)
        .resize_filter(filter)
        .to_hasher();

    let hash = hasher.hash_image(&img);
    let flip_hash = hasher.hash_image(&img.flipv());
    let flop_hash = hasher.hash_image(&img.fliph());

    Ok(HashResult {
        key: job.key.clone(),
        hash: B64.encode(hash.as_bytes()),
        flip_hash: B64.encode(flip_hash.as_bytes()),
        flop_hash: B64.encode(flop_hash.as_bytes()),
        width,
        height,
    })
}
