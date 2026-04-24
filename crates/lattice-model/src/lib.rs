//! Model source handling and zero-copy weight mapping.

pub mod gguf;
pub mod llama;

use std::fs::File;
use std::path::{Path, PathBuf};

pub use gguf::{
    ArchitectureMetadata, GgmlType, GgmlTypeLayout, GgufFile, GgufHeader, GgufMetadata,
    GgufMetadataValue, GgufMetadataValueType, GgufTensorInfo, GgufTensorView, LlamaMetadata,
    parse_gguf, parse_gguf_metadata,
};
use lattice_core::{LatticeError, Result};
pub use llama::{LlamaBlockTensors, LlamaModel, LlamaModelSpec, LlamaTensorMap};
use memmap2::{Mmap, MmapOptions};

/// Supported weight file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// GGUF weights.
    Gguf,
    /// Safetensors weights.
    Safetensors,
}

impl WeightFormat {
    /// Detects the weight format based on the path extension.
    pub fn detect(path: &Path) -> Result<Self> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("gguf") => Ok(Self::Gguf),
            Some("safetensors") => Ok(Self::Safetensors),
            _ => Err(LatticeError::UnsupportedModelFormat(path.to_path_buf())),
        }
    }
}

/// Metadata about a model file before mapping.
#[derive(Debug, Clone)]
pub struct ModelSource {
    path: PathBuf,
    format: WeightFormat,
}

impl ModelSource {
    /// Constructs a source from a local filesystem path.
    pub fn from_path(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let format = WeightFormat::detect(&path)?;
        Ok(Self { path, format })
    }

    /// Returns the local path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the detected weight format.
    pub const fn format(&self) -> WeightFormat {
        self.format
    }
}

/// Memory-mapped model weights.
#[derive(Debug)]
pub struct MappedWeights {
    source: ModelSource,
    mmap: Mmap,
}

impl MappedWeights {
    /// Maps a model file from disk.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let source = ModelSource::from_path(path)?;
        let file = File::open(source.path())?;
        // SAFETY: The mapped file handle remains valid for the duration of the map creation.
        let mmap = unsafe { MmapOptions::new().map(&file) }?;

        Ok(Self { source, mmap })
    }

    /// Returns the mapped byte length.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns whether the mapped file is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Returns the model source metadata.
    pub fn source(&self) -> &ModelSource {
        &self.source
    }

    /// Parses GGUF metadata from the mapped model file.
    pub fn gguf_metadata(&self) -> Result<GgufMetadata> {
        if self.source.format() != WeightFormat::Gguf {
            return Err(LatticeError::Message(format!(
                "cannot parse GGUF metadata from {:?} weights",
                self.source.format()
            )));
        }

        parse_gguf_metadata(self.as_bytes())
    }

    /// Parses the mapped GGUF file and enables zero-copy tensor lookup.
    pub fn gguf_file(&self) -> Result<GgufFile<'_>> {
        if self.source.format() != WeightFormat::Gguf {
            return Err(LatticeError::Message(format!(
                "cannot parse GGUF file from {:?} weights",
                self.source.format()
            )));
        }

        parse_gguf(self.as_bytes())
    }

    /// Returns the mapped bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::WeightFormat;

    #[test]
    fn detect_supported_formats() {
        assert_eq!(
            WeightFormat::detect(Path::new("model.gguf")).expect("GGUF should be supported"),
            WeightFormat::Gguf
        );
        assert_eq!(
            WeightFormat::detect(Path::new("model.safetensors"))
                .expect("Safetensors should be supported"),
            WeightFormat::Safetensors
        );
    }
}
