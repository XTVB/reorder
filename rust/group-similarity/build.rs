fn main() {
    // Link Apple Accelerate framework for BLAS (cblas_sgemm)
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
