// src/assets/file.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// References a source file by path.
///
/// For supported kinds the build compiles the file into the world (an `.obj`
/// becomes mesh data); other kinds are path-only references.
#[derive(Debug)]
pub struct File {
    pub asset_id: AssetId,
    pub path: String,
    pub kind: Option<FileKind>,
    /// Injected at load time for kinds that produce a compiled blob (e.g. obj → mesh payload).
    pub locator: Option<PayloadLocator>,
}

impl Component for File {
    const NAME: &'static str = "File";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;

    type Args = FileArgs;

    fn to_args(&self) -> FileArgs {
        FileArgs {
            path: self.path.clone(),
            kind: self.kind.clone(),
        }
    }

    fn from_args(args: FileArgs) -> Self {
        let kind = args.kind.clone().or_else(|| {
            std::path::Path::new(&args.path)
                .extension()
                .and_then(|e| e.to_str())
                .and_then(FileKind::from_ext)
        });
        Self {
            asset_id: AssetId::default(),
            path: args.path,
            kind,
            locator: None,
        }
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::build::SourceBacked for File {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

/// The category of file content, inferred from the extension when not supplied.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileKind {
    Obj,
    Png,
    Jpg,
    Jpeg,
    Bmp,
    Tga,
    Gif,
    Ttf,
    Otf,
    Txt,
    Md,
    Mtl,
}

impl FileKind {
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "obj" => Some(Self::Obj),
            "png" => Some(Self::Png),
            "jpg" => Some(Self::Jpg),
            "jpeg" => Some(Self::Jpeg),
            "bmp" => Some(Self::Bmp),
            "tga" => Some(Self::Tga),
            "gif" => Some(Self::Gif),
            "ttf" => Some(Self::Ttf),
            "otf" => Some(Self::Otf),
            "txt" => Some(Self::Txt),
            "md" => Some(Self::Md),
            "mtl" => Some(Self::Mtl),
            _ => None,
        }
    }

    /// Returns true for kinds whose build output is a mesh blob compatible with the
    /// mesh payload format (vertex + index data readable by GraphicsSystem).
    pub fn is_mesh(&self) -> bool {
        matches!(self, Self::Obj)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct FileArgs {
    /// Path to the source file, relative to the project root.
    pub path: String,
    /// File category. Inferred from the path extension when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<FileKind>,
}
