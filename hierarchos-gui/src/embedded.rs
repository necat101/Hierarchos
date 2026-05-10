// embedded.rs — Embeds the Hierarchos Python package into the binary
//
// At build time, all Python source files are embedded via include_str!().
// At runtime, extract_embedded_python() writes them to the app data directory
// so the bridge server can import them.

use std::fs;
use std::path::{Path, PathBuf};

/// Embedded Python source files.
/// Each entry: (relative_path, file_contents)
const EMBEDDED_FILES: &[(&str, &str)] = &[
    // Bridge server
    (
        "hierarchos_bridge_server.py",
        include_str!("../../hierarchos_bridge_server.py"),
    ),
    // Package root
    (
        "hierarchos/__init__.py",
        include_str!("../../hierarchos/__init__.py"),
    ),
    // Models
    (
        "hierarchos/models/core.py",
        include_str!("../../hierarchos/models/core.py"),
    ),
    (
        "hierarchos/models/ltm.py",
        include_str!("../../hierarchos/models/ltm.py"),
    ),
    (
        "hierarchos/models/rwkv_cell.py",
        include_str!("../../hierarchos/models/rwkv_cell.py"),
    ),
    (
        "hierarchos/models/quantized.py",
        include_str!("../../hierarchos/models/quantized.py"),
    ),
    // Training
    (
        "hierarchos/training/trainer.py",
        include_str!("../../hierarchos/training/trainer.py"),
    ),
    (
        "hierarchos/training/datasets.py",
        include_str!("../../hierarchos/training/datasets.py"),
    ),
    (
        "hierarchos/training/optimizers.py",
        include_str!("../../hierarchos/training/optimizers.py"),
    ),
    // Inference
    (
        "hierarchos/inference/chat.py",
        include_str!("../../hierarchos/inference/chat.py"),
    ),
    // Utils
    (
        "hierarchos/utils/device.py",
        include_str!("../../hierarchos/utils/device.py"),
    ),
    (
        "hierarchos/utils/checkpoint.py",
        include_str!("../../hierarchos/utils/checkpoint.py"),
    ),
    (
        "hierarchos/utils/rosa.py",
        include_str!("../../hierarchos/utils/rosa.py"),
    ),
    // Evaluation
    (
        "hierarchos/evaluation/__init__.py",
        include_str!("../../hierarchos/evaluation/__init__.py"),
    ),
    (
        "hierarchos/evaluation/evaluator.py",
        include_str!("../../hierarchos/evaluation/evaluator.py"),
    ),
    (
        "hierarchos/evaluation/lm_eval_wrapper.py",
        include_str!("../../hierarchos/evaluation/lm_eval_wrapper.py"),
    ),
];

/// A simple version hash based on total content length.
/// Changes when any embedded file is modified.
fn content_hash() -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for (path, content) in EMBEDDED_FILES {
        hasher.update(path.as_bytes());
        hasher.update(content.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Get the app data directory for Hierarchos.
pub fn get_app_data_dir() -> PathBuf {
    if let Some(proj_dirs) = directories::ProjectDirs::from("com", "hierarchos", "hierarchos-gui") {
        proj_dirs.data_local_dir().to_path_buf()
    } else {
        // Fallback to a directory next to the executable
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
            .join("hierarchos_data")
    }
}

/// Extract all embedded Python files to the given directory.
/// Returns the path to `hierarchos_bridge_server.py`.
///
/// Files are only re-extracted if the version hash has changed,
/// making subsequent launches instant.
pub fn extract_embedded_python() -> Result<PathBuf, String> {
    let base_dir = get_app_data_dir().join("python");
    let hash_file = base_dir.join(".version_hash");
    let current_hash = content_hash();

    // Check if already extracted with same version
    if hash_file.exists() {
        if let Ok(existing) = fs::read_to_string(&hash_file) {
            if existing.trim() == current_hash {
                let server_path = base_dir.join("hierarchos_bridge_server.py");
                if server_path.exists() {
                    return Ok(server_path);
                }
            }
        }
    }

    // Extract all files
    for (rel_path, content) in EMBEDDED_FILES {
        let full_path = base_dir.join(rel_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {:?}: {}", parent, e))?;
        }

        // Also create empty __init__.py for intermediate packages
        // (e.g. hierarchos/models/__init__.py)
        if let Some(parent) = full_path.parent() {
            let init = parent.join("__init__.py");
            if !init.exists() && parent != base_dir {
                let _ = fs::write(&init, "");
            }
        }

        fs::write(&full_path, content)
            .map_err(|e| format!("Failed to write {:?}: {}", full_path, e))?;
    }

    // Write version hash
    let _ = fs::write(&hash_file, &current_hash);

    Ok(base_dir.join("hierarchos_bridge_server.py"))
}

/// Get the extraction directory (for PYTHONPATH).
pub fn get_python_base_dir() -> PathBuf {
    get_app_data_dir().join("python")
}

/// Get the default model download directory.
pub fn get_models_dir() -> PathBuf {
    get_app_data_dir().join("models")
}
