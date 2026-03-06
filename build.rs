fn main() {
    // vec0.so ships without the conventional "lib" prefix, so we can't use
    // -lvec0.  Pass the full path directly as a linker argument instead.
    //
    // Use SQLITE_VEC_PATH env var if set, otherwise search common locations.
    let path = std::env::var("SQLITE_VEC_PATH").unwrap_or_else(|_| {
        let candidates = [
            "/usr/lib/vec0.so",
            "/usr/local/lib/vec0.so",
            "/usr/lib64/vec0.so",
            "/usr/lib/x86_64-linux-gnu/vec0.so",
            "/usr/lib/aarch64-linux-gnu/vec0.so",
        ];
        for c in &candidates {
            if std::path::Path::new(c).exists() {
                return c.to_string();
            }
        }
        // Fall back to /usr/lib and let the linker error if missing.
        "/usr/lib/vec0.so".to_string()
    });

    println!("cargo:rerun-if-env-changed=SQLITE_VEC_PATH");
    println!("cargo:rustc-link-arg={path}");
}
