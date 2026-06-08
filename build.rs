use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=models/manifest.json");
    println!("cargo:rerun-if-env-changed=REQUIRE_TRANSLATION_MODELS");
    println!("cargo:rerun-if-env-changed=TRANSLATION_MODEL_DIR");

    validate_manifest();

    validate_feature_selection();

    if cfg!(feature = "translation-ct2-cuda") {
        warn(
            "translation-ct2-cuda is enabled; release packaging must include CUDA-compatible CTranslate2/runtime libraries if they are not provided by the system",
        );
    }
}

fn validate_manifest() {
    let manifest_path = PathBuf::from("models/manifest.json");

    match fs::read_to_string(&manifest_path) {
        Ok(contents) => {
            if contents.trim().is_empty() {
                warn(&format!(
                    "translation model manifest {} is empty",
                    manifest_path.display()
                ));
                maybe_fail("translation model manifest is empty");
            }
        }
        Err(error) => {
            warn(&format!(
                "translation model manifest {} is not readable: {error}",
                manifest_path.display()
            ));
            maybe_fail("translation model manifest is required");
        }
    }
}

fn validate_feature_selection() {
    if cfg!(all(
        feature = "translation-ct2-cpu",
        feature = "translation-ct2-cuda"
    )) {
        panic!(
            "features translation-ct2-cpu and translation-ct2-cuda are both enabled; build CPU and CUDA release artifacts separately"
        );
    }
}

fn env_flag(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn maybe_fail(message: &str) {
    if env_flag("REQUIRE_TRANSLATION_MODELS") {
        panic!("{message}");
    }
}

fn warn(message: &str) {
    println!("cargo:warning={message}");
}
