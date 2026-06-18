// asset_impls/audio_clip.rs

use concinnity_core::assets::AudioClip;

impl crate::asset::BuildAsset for AudioClip {
    fn compile_payload(
        args: &serde_json::Value,
        _ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::audio_clip::compile_audio_clip_payload(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
