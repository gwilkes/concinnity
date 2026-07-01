// examples/bistro/main.rs
//
// Standalone host for the Amazon Lumberyard Bistro showcase. On first run it
// fetches the Bistro asset pack into examples/bistro/assets/ (~833 MB), then
// compiles the sibling world.jsonl in memory and plays it through the runtime
// renderer. Subsequent runs find the assets already present and skip the fetch.
//
// The fetch is a runtime preflight, not a build step: `cargo build` never
// touches the network, and the download happens once, the first time someone
// runs the example.
//
// The renderer is heavy here (2.8M triangles, ray-traced reflections, SSGI).
// Run in release for full frame rate: `cargo run -p bistro --release`.

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use example_common::{compile_world, init_logging, paths, run};

// The NVIDIA ORCA download. It redirects to the actual archive; ureq follows
// redirects. Override with BISTRO_URL, or point BISTRO_ARCHIVE at an
// already-downloaded archive to skip the download entirely.
const BISTRO_URL: &str = "https://developer.nvidia.com/bistro";

// Sentinel files: if both already exist, the assets are considered present and
// the fetch is skipped. Both ship inside the same archive. Paths are relative
// to this example's directory (main chdirs there before anything else).
const FBX_REL: &str = "assets/Bistro_v5_2/BistroExterior.fbx";
const HDR_REL: &str = "assets/Bistro_v5_2/san_giuseppe_bridge_4k.hdr";

// Windows' system `d3d12.dll` reads these two symbols from the host EXE's PE
// export table at process start: when both are present and the named SDK path
// resolves to a directory containing `D3D12Core.dll`, it loads that copy in
// place of the OS-bundled (older) D3D12 runtime. Modern FidelityFX FSR3 needs
// the Agility SDK; without these exports `ffxCreateContext` throws a C++
// exception that aborts the process. The companion build.rs copies the Agility
// DLLs into `target/{profile}/D3D12/` (via concinnity-toolchain) and emits
// the matching linker exports, so a final binary needs both. The version value
// must match the NuGet package: the directory name is
// `microsoft.direct3d.d3d12.1.<VER>.<PATCH>`. Mirrors concinnity-editor's
// main.rs; keep the two in sync when bumping the Agility SDK.
//
// `#[used]` forces the linker to keep the symbols even though nothing in Rust
// references them; `#[no_mangle]` keeps the exact case-sensitive name
// `d3d12.dll` looks up.
#[cfg(backend_dx)]
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKVersion: u32 = 619;

#[cfg(backend_dx)]
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKPath: &[u8; 9] = b".\\D3D12\\\0";

fn main() -> io::Result<()> {
    // fbxcel logs a benign WARN about an "extra node end marker" near the end of
    // BistroExterior.fbx -- a known quirk of that file's binary node
    // terminators. The parse recovers and the whole scene imports, so silence
    // just that crate while leaving every other log at its normal level. The
    // runtime's default filter is info (debug) / warn (release); mirror it and
    // append the fbxcel directive, but only when the user hasn't set RUST_LOG.
    if std::env::var_os("RUST_LOG").is_none() {
        let base = if cfg!(debug_assertions) {
            "info"
        } else {
            "warn"
        };
        unsafe { std::env::set_var("RUST_LOG", format!("{base},fbxcel=error")) };
    }

    init_logging();

    // Anchor the `.concinnity/` state tree (payload cache, runtime config) to
    // wherever the command was invoked before the chdir below moves the working
    // directory. Without this, chdir'ing into the example directory would create
    // `.concinnity/` there instead of at the invocation point.
    if let Ok(invocation_dir) = std::env::current_dir() {
        paths::set_root(invocation_dir);
    }

    // Resolve every relative asset path in world.jsonl against this example's
    // own directory rather than wherever cargo was invoked from.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::env::set_current_dir(&dir)
        .map_err(|e| io::Error::new(e.kind(), format!("could not enter {}: {e}", dir.display())))?;

    ensure_assets()?;

    if cfg!(debug_assertions) {
        eprintln!("note: debug build -- run `cargo run -p bistro --release` for full frame rate");
    }

    let content = std::fs::read_to_string("world.jsonl")?;
    run(compile_world(&content)?)
}

// Download and unpack the Bistro asset pack unless it is already on disk.
fn ensure_assets() -> io::Result<()> {
    if assets_present(Path::new(".")) {
        return Ok(());
    }

    let archive = match std::env::var("BISTRO_ARCHIVE") {
        Ok(local) => {
            eprintln!("using local Bistro archive: {local}");
            PathBuf::from(local)
        }
        Err(_) => {
            let url = std::env::var("BISTRO_URL").unwrap_or_else(|_| BISTRO_URL.to_string());
            eprintln!(
                "Bistro assets not found. Downloading into \
                 examples/bistro/assets/ from {url}"
            );
            let tmp = std::env::temp_dir().join("concinnity_bistro_download");
            download(&url, &tmp)?;
            tmp
        }
    };

    let mut header = [0u8; 4];
    {
        let mut f = std::fs::File::open(&archive)?;
        f.read_exact(&mut header)?;
    }
    if !looks_like_zip(&header) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "downloaded archive is not a ZIP (first bytes {header:02x?}). The download URL \
                 may have changed format; fetch the pack manually and set BISTRO_ARCHIVE to it \
                 (or place the result under examples/bistro/assets/)."
            ),
        ));
    }

    eprintln!("extracting into examples/bistro/assets/ ...");
    std::fs::create_dir_all("assets")?;
    extract_zip(&archive, Path::new("assets"))?;

    // Only the download path created the temp file; a user-supplied archive is
    // left alone.
    if std::env::var("BISTRO_ARCHIVE").is_err() {
        let _ = std::fs::remove_file(&archive);
    }

    if !assets_present(Path::new(".")) {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "extraction finished but the expected files are still missing; the archive layout \
             may differ from assets/Bistro_v5_2/.",
        ));
    }

    eprintln!("Bistro assets ready.");
    Ok(())
}

// Stream a URL to a file, printing coarse progress. Streams to disk rather than
// memory because the archive is gigabytes.
fn download(url: &str, dest: &Path) -> io::Result<()> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| io::Error::other(format!("request failed: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        return Err(io::Error::other(format!("server returned HTTP {status}")));
    }

    let total: Option<u64> = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok());

    let mut reader = resp.into_body().into_reader();
    let mut file = std::fs::File::create(dest)?;

    let mut buf = vec![0u8; 1 << 20];
    let mut downloaded: u64 = 0;
    let mut last_report: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        if downloaded - last_report >= 64 << 20 {
            report_progress(downloaded, total);
            last_report = downloaded;
        }
    }
    report_progress(downloaded, total);
    eprintln!();
    Ok(())
}

fn report_progress(downloaded: u64, total: Option<u64>) {
    let mib = |b: u64| b as f64 / (1 << 20) as f64;
    match total {
        Some(t) if t > 0 => eprint!(
            "\r  downloaded {:.0} / {:.0} MiB ({:.0}%)   ",
            mib(downloaded),
            mib(t),
            downloaded as f64 / t as f64 * 100.0
        ),
        _ => eprint!("\r  downloaded {:.0} MiB   ", mib(downloaded)),
    }
    let _ = io::stderr().flush();
}

// Unpack a ZIP archive into a destination directory, preserving its internal
// paths (the archive carries a top-level Bistro_v5_2/ folder).
fn extract_zip(archive: &Path, dest: &Path) -> io::Result<()> {
    let file = std::fs::File::open(archive)?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("not a valid zip: {e}")))?;
    zip.extract(dest)
        .map_err(|e| io::Error::other(format!("extract failed: {e}")))?;
    Ok(())
}

// True when an archive begins with the ZIP local-file-header magic ("PK\x03\x04").
fn looks_like_zip(header: &[u8]) -> bool {
    header.starts_with(&[0x50, 0x4b, 0x03, 0x04])
}

// True when both sentinel files exist under `root`.
fn assets_present(root: &Path) -> bool {
    root.join(FBX_REL).is_file() && root.join(HDR_REL).is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_magic_is_recognised() {
        assert!(looks_like_zip(&[0x50, 0x4b, 0x03, 0x04, 0x14]));
        // gzip and tar magics, and a short slice, are not ZIPs.
        assert!(!looks_like_zip(&[0x1f, 0x8b, 0x08, 0x00]));
        assert!(!looks_like_zip(&[0x50, 0x4b]));
        assert!(!looks_like_zip(&[]));
    }

    #[test]
    fn assets_present_requires_both_sentinels() {
        let root =
            std::env::temp_dir().join(format!("concinnity_bistro_test_{}", std::process::id()));
        let bistro = root.join("assets/Bistro_v5_2");
        std::fs::create_dir_all(&bistro).unwrap();

        assert!(!assets_present(&root), "no files yet");

        std::fs::write(bistro.join("BistroExterior.fbx"), b"x").unwrap();
        assert!(!assets_present(&root), "fbx alone is not enough");

        std::fs::write(bistro.join("san_giuseppe_bridge_4k.hdr"), b"x").unwrap();
        assert!(assets_present(&root), "both sentinels present");

        std::fs::remove_dir_all(&root).unwrap();
    }
}
