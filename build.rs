fn main() {
    // vec0.so ships without the conventional "lib" prefix, so we can't use
    // -lvec0.  Pass the full path directly as a linker argument instead.
    println!("cargo:rustc-link-arg=/usr/lib/vec0.so");
}
