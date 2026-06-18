// src/result.rs
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CnResult {
    #[error("Success")]
    #[allow(dead_code)]
    Success = 0,

    #[error("Invalid asset type")]
    AssetInvalidType,

    // Generic
    #[error("Invalid state")]
    InvalidState,
    #[error("Invalid argument")]
    InvalidArgument,

    #[error("File I/O error")]
    FileIo,
}

impl From<std::io::Error> for CnResult {
    fn from(_e: std::io::Error) -> Self {
        CnResult::FileIo
    }
}

impl From<serde_json::Error> for CnResult {
    fn from(e: serde_json::Error) -> Self {
        tracing::error!("JSON deserialization error: {}", e);
        CnResult::InvalidArgument // or better: CnResult::JsonError (see note below)
    }
}
