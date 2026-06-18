// src/build/audio_clip.rs
//
// Compile an AudioClip's source file into a blob payload. The payload is the
// encoded audio file bytes verbatim; the runtime decodes them with kira. The
// build decodes the file once and discards the result, so a clip the engine
// cannot read fails the build instead of the render loop.

use std::io::Cursor;

use kira::sound::static_sound::StaticSoundData;

// Read and validate the audio file named by `args["source"]`, returning its
// bytes for the world blob.
pub(crate) fn compile_audio_clip_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let source = args
        .get("source")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "AudioClip: missing 'source'".to_string())?;
    let bytes =
        std::fs::read(source).map_err(|e| format!("AudioClip: failed to read '{source}': {e}"))?;
    // Decode-and-discard: a clip the engine cannot decode fails the build.
    StaticSoundData::from_cursor(Cursor::new(bytes.clone()))
        .map_err(|e| format!("AudioClip: '{source}' is not a decodable audio file: {e}"))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // Build a minimal 16-bit mono PCM WAV (0.1 s of silence at 8 kHz) so the
    // happy path can be exercised without a checked-in binary fixture.
    fn tiny_wav() -> Vec<u8> {
        let sample_rate: u32 = 8000;
        let samples: u32 = sample_rate / 10;
        let data_len = samples * 2; // 16-bit mono
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend(std::iter::repeat_n(0u8, data_len as usize));
        wav
    }

    #[test]
    fn missing_source_errors() {
        assert!(compile_audio_clip_payload(&serde_json::json!({})).is_err());
        assert!(compile_audio_clip_payload(&serde_json::json!({ "source": "" })).is_err());
    }

    #[test]
    fn nonexistent_file_errors() {
        let args = serde_json::json!({ "source": "definitely_not_here_9182.ogg" });
        assert!(compile_audio_clip_payload(&args).is_err());
    }

    #[test]
    fn non_audio_bytes_are_rejected() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"this is plainly not an audio file")
            .unwrap();
        let args = serde_json::json!({ "source": file.path().to_str().unwrap() });
        assert!(compile_audio_clip_payload(&args).is_err());
    }

    #[test]
    fn valid_wav_compiles_to_its_own_bytes() {
        let wav = tiny_wav();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&wav).unwrap();
        let args = serde_json::json!({ "source": file.path().to_str().unwrap() });
        let payload = compile_audio_clip_payload(&args).expect("valid WAV should compile");
        assert_eq!(payload, wav, "payload should be the source bytes verbatim");
    }
}
