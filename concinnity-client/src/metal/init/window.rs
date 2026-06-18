// src/metal/init/window.rs
//
// NSWindow + MTKView setup for MtlContext::new, plus the initial HDR target
// sizing decision (geometry-less worlds clamp to 1x1; otherwise the drawable
// size wins, falling back to the requested width/height before the drawable
// exists).
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::MainThreadOnly;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::{
    NSApplication, NSAutoresizingMaskOptions, NSBackingStoreType, NSScreen, NSView, NSWindow,
    NSWindowStyleMask,
};
use objc2_core_graphics::{
    CGColorSpace, kCGColorSpaceDisplayP3_PQ, kCGColorSpaceExtendedLinearDisplayP3,
};
use objc2_foundation::{NSPoint, NSRect, NSSize};
use objc2_metal::{MTLDevice, MTLPixelFormat};
use objc2_metal_kit::MTKView;
use objc2_quartz_core::CAMetalLayer;

use crate::gfx::hdr_output::HdrOutputMode;
use crate::metal::context::{take_embedded_pump_events, take_embedded_view};
use crate::metal::pipeline::ns_str;

pub(crate) struct WindowSetup {
    pub window: Option<Retained<NSWindow>>,
    pub mtk_view: Retained<MTKView>,
    pub pump_events: bool,
    pub initial_w: u32,
    pub initial_h: u32,
    // Shared native-fullscreen flag, kept in sync by `window_delegate`. False
    // in embedded mode (no NSWindow).
    pub fullscreen: std::sync::Arc<std::sync::atomic::AtomicBool>,
    // NSWindowDelegate tracking the fullscreen transition; None in embedded
    // mode. The caller stores it so the window's weak delegate stays attached.
    pub window_delegate: Option<Retained<crate::metal::window_delegate::WindowDelegate>>,
    // Resolved swapchain colour-output mode. `Sdr` when the world did not
    // request HDR or the active display lacks EDR headroom; `Hdr` when the
    // CAMetalLayer was configured with `RGBA16Float` + extended-linear
    // Display P3 colour space + `wantsExtendedDynamicRangeContent = true`.
    pub hdr_mode: HdrOutputMode,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_window_and_view(
    mtm: objc2::MainThreadMarker,
    device: &ProtocolObject<dyn MTLDevice>,
    title: &str,
    width: u32,
    height: u32,
    geometry_less: bool,
    hdr_display_requested: bool,
    hdr_pq_requested: bool,
    capture_enabled: bool,
) -> Result<WindowSetup, String> {
    // Resolve the swapchain colour-output mode. EDR support is per-display,
    // so the answer depends on which screen the window will land on. In
    // windowed mode we use `NSWindow::screen()` after attaching; in embedded
    // mode (preview) we fall back to the main screen since the parent NSView
    // does not always have a window at this point. The asset toggle is the
    // outer gate: a world that did not opt in stays SDR even on a capable
    // panel.
    let max_edr = measure_max_edr(mtm);
    let hdr_mode = HdrOutputMode::resolve(hdr_display_requested, hdr_pq_requested, max_edr);
    if hdr_display_requested && !hdr_mode.is_hdr() {
        tracing::warn!(
            "HDR display requested but the active display reports max EDR \
             multiplier {:.3}: falling back to SDR (BGRA8Unorm) output",
            max_edr
        );
    } else if let HdrOutputMode::Hdr { encoding, .. } = hdr_mode {
        tracing::info!(
            "HDR display output enabled: max EDR multiplier {:.3}, encoding={:?}",
            max_edr,
            encoding,
        );
    }

    let embedded_ptr = take_embedded_view();
    // Windowed mode always pumps events (CLI behaviour). Embedded mode is
    // quiet by default; the play-in-view path opts in via set_embedded_pump_events.
    let pump_events = embedded_ptr.is_null() || take_embedded_pump_events();
    let (window, mtk_view, fullscreen, window_delegate) = if embedded_ptr.is_null() {
        // Windowed mode: create a new NSWindow containing the MTKView.
        let window = create_window(mtm, title, width, height)?;
        let content_rect = window.contentRectForFrameRect(window.frame());
        let mtk_view =
            MTKView::initWithFrame_device(MTKView::alloc(mtm), content_rect, Some(device));
        configure_mtk_view(&mtk_view, hdr_mode, capture_enabled);
        window.setContentView(Some(&mtk_view));
        // Track native-fullscreen state authoritatively (the style-mask bit
        // lags the animated transition) so the settings menu's Window Mode row
        // never toggles the wrong way.
        let (delegate, fullscreen) =
            crate::metal::window_delegate::attach_fullscreen_delegate(mtm, &window);
        NSApplication::sharedApplication(mtm).activate();
        window.makeKeyAndOrderFront(None);
        (Some(window), mtk_view, fullscreen, Some(delegate))
    } else {
        // Embedded mode: attach an MTKView as a subview of the provided NSView.
        // The Swift caller owns the parent NSView and keeps it alive for the
        // preview lifetime; we borrow a raw reference here only during init.
        let parent: &NSView = unsafe { &*(embedded_ptr as *const NSView) };
        let bounds = parent.bounds();
        let mtk_view = MTKView::initWithFrame_device(MTKView::alloc(mtm), bounds, Some(device));
        configure_mtk_view(&mtk_view, hdr_mode, capture_enabled);
        mtk_view.setAutoresizingMask(
            NSAutoresizingMaskOptions::ViewWidthSizable
                | NSAutoresizingMaskOptions::ViewHeightSizable,
        );
        parent.addSubview(&mtk_view);
        // No NSWindow we own in embedded mode, so no fullscreen delegate; the
        // flag stays false (set_window_mode is a no-op without self.window).
        (
            None,
            mtk_view,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            None,
        )
    };

    // Initial HDR target sizing. The drawable may not exist yet (especially
    // in embedded mode before the parent view finishes layout), so we use
    // the requested width/height as a starting size and let draw_frame
    // resize the targets if the actual drawable size differs.
    let initial_drawable_size = mtk_view.drawableSize();
    // A geometry-less world (e.g. text-only) renders no 3D content into the
    // off-screen HDR / bloom / effect targets, so they are allocated at 1x1
    // rather than full resolution -- this avoids paying for a full MSAA HDR
    // colour + depth + bloom chain (tens of MB) for a trivial 2D world.
    let initial_w = if geometry_less {
        1
    } else if initial_drawable_size.width > 0.0 {
        initial_drawable_size.width as u32
    } else {
        width.max(1)
    };
    let initial_h = if geometry_less {
        1
    } else if initial_drawable_size.height > 0.0 {
        initial_drawable_size.height as u32
    } else {
        height.max(1)
    };

    Ok(WindowSetup {
        window,
        mtk_view,
        pump_events,
        initial_w,
        initial_h,
        fullscreen,
        window_delegate,
        hdr_mode,
    })
}

// Largest extended-range colour-component multiplier the system thinks any
// attached screen can drive. SDR panels report `1.0`; HDR panels report
// `2.0`+. With no screens at all (a head-less unit test or detached embedded
// preview) the function returns `1.0` so the resolver stays on the SDR
// path.
//
// We query the *potential* headroom rather than the *current* headroom
// because `measure_max_edr` runs before the CAMetalLayer is configured for
// EDR: at that point macOS hasn't yet allocated any HDR headroom for our
// window, so `maximumExtendedDynamicRangeColorComponentValue` returns the
// idle value (commonly `1.0`) regardless of the panel's capabilities. The
// `Potential` API reports what the panel CAN do once HDR content is on
// screen, which is what we need at gate time. A separate live readout
// could later poll the dynamic value each frame for an in-game brightness
// monitor; the static gate only needs to know whether HDR is a possibility.
pub(crate) fn measure_max_edr(mtm: objc2::MainThreadMarker) -> f32 {
    let screens = NSScreen::screens(mtm);
    let mut best: f64 = 1.0;
    for i in 0..screens.count() {
        let s = screens.objectAtIndex(i);
        let v = s.maximumPotentialExtendedDynamicRangeColorComponentValue();
        if v > best {
            best = v;
        }
    }
    best as f32
}

// The drawable is now the composite/post-pass target only -- main pass renders
// into an off-screen RGBA16Float MSAA target with its own depth. The drawable
// therefore has no depth attachment and no MSAA.
//
// In HDR mode the swapchain colour attachment is widened from BGRA8Unorm to
// RGBA16Float and the underlying CAMetalLayer is reconfigured for extended
// dynamic-range output: extended-linear Display P3 colour space, EDR content
// enabled. The post-process fragment then writes linear extended-range values
// straight through (no tonemap / gamma).
fn configure_mtk_view(mtk_view: &MTKView, hdr_mode: HdrOutputMode, capture_enabled: bool) {
    mtk_view.setPaused(true);
    mtk_view.setEnableSetNeedsDisplay(false);
    let swap_fmt = swap_pixel_format(hdr_mode);
    mtk_view.setColorPixelFormat(swap_fmt);
    mtk_view.setDepthStencilPixelFormat(MTLPixelFormat::Invalid);
    mtk_view.setSampleCount(1);
    // The drawable defaults to `framebufferOnly` (write-as-attachment only),
    // which forbids using it as a blit source. Under the capture-enabled
    // (`cn debug`) path the `screenshot` command blits the last presented
    // drawable back to the host, so switch it off there. Left at the default
    // in production so a normal `cn run` pays nothing for a debug-only feature.
    if capture_enabled {
        mtk_view.setFramebufferOnly(false);
    }

    if let HdrOutputMode::Hdr { encoding, .. } = hdr_mode {
        configure_hdr_layer(mtk_view, encoding);
    }
}

// Pixel format the swapchain attachment uses. BGRA8Unorm is the historical
// SDR default; RGBA16Float gives the EDR path the headroom to drive values
// past SDR reference white without crushing precision.
pub(crate) fn swap_pixel_format(hdr_mode: HdrOutputMode) -> MTLPixelFormat {
    if hdr_mode.is_hdr() {
        MTLPixelFormat::RGBA16Float
    } else {
        MTLPixelFormat::BGRA8Unorm
    }
}

// Turn display sync (vsync) on or off on the MTKView's backing CAMetalLayer.
// `displaySyncEnabled` true locks presentation to the display refresh; false
// presents as soon as a frame is ready (uncapped, possible tearing). Used both
// at init (to honor GraphicsConfig.vsync) and at runtime (settings menu). A nil
// or non-CAMetalLayer layer is treated as a no-op, matching configure_hdr_layer.
pub(crate) fn set_display_sync(mtk_view: &MTKView, on: bool) {
    let Some(layer) = mtk_view.layer() else {
        return;
    };
    if let Some(metal_layer) = layer.downcast_ref::<CAMetalLayer>() {
        metal_layer.setDisplaySyncEnabled(on);
    }
}

// Apply EDR layer flags to the MTKView's backing CAMetalLayer. MTKView's
// `layer` property is documented to be a CAMetalLayer, but `NSView::layer()`
// returns the parent CALayer type: we cast through the runtime-safe `cast`
// path and only flip the EDR + colour-space switches once we have it. A nil
// layer is reported and treated as a no-op (the renderer then falls back to
// the standard sRGB SDR path silently; we already log warn-level above when
// EDR is requested but not achievable).
fn configure_hdr_layer(mtk_view: &MTKView, encoding: crate::gfx::hdr_output::HdrEncoding) {
    let Some(layer) = mtk_view.layer() else {
        tracing::warn!(
            "HDR display requested but MTKView has no backing layer: falling back to SDR"
        );
        return;
    };
    // The MTKView documents its layer is always CAMetalLayer. The runtime
    // class check inside `downcast_ref` keeps this safe in the unlikely event
    // that a future MTKView build hands us something else.
    let metal_layer: &CAMetalLayer = match layer.downcast_ref::<CAMetalLayer>() {
        Some(l) => l,
        None => {
            tracing::warn!(
                "HDR display requested but the MTKView layer is not a CAMetalLayer: falling \
                 back to SDR"
            );
            return;
        }
    };
    // Pick the swapchain colour space by encoding:
    //   - ExtendedLinear → kCGColorSpaceExtendedLinearDisplayP3. The shader
    //     writes linear values where `1.0` is SDR reference white; the
    //     compositor handles the panel-side encode.
    //   - Pq → kCGColorSpaceDisplayP3_PQ. The shader emits PQ-encoded values
    //     directly; the panel decodes via the PQ EOTF. Same Display P3
    //     primaries as the linear path so the gamut situation is unchanged.
    let (name, label): (_, &str) = match encoding {
        crate::gfx::hdr_output::HdrEncoding::ExtendedLinear => (
            unsafe { kCGColorSpaceExtendedLinearDisplayP3 },
            "kCGColorSpaceExtendedLinearDisplayP3",
        ),
        crate::gfx::hdr_output::HdrEncoding::Pq => (
            unsafe { kCGColorSpaceDisplayP3_PQ },
            "kCGColorSpaceDisplayP3_PQ",
        ),
    };
    let colorspace = CGColorSpace::with_name(Some(name));
    match colorspace.as_deref() {
        Some(cs) => metal_layer.setColorspace(Some(cs)),
        None => tracing::warn!(
            "{} unavailable: leaving CAMetalLayer at default colour space (HDR output may \
             look desaturated)",
            label
        ),
    }
    metal_layer.setWantsExtendedDynamicRangeContent(true);
    // CAMetalLayer.pixelFormat is normally set by MTKView::setColorPixelFormat
    // above, but make it explicit so a future refactor that drops the MTKView
    // hop does not silently bring the layer back to BGRA8Unorm.
    metal_layer.setPixelFormat(MTLPixelFormat::RGBA16Float);
}

pub(crate) fn create_window(
    mtm: objc2::MainThreadMarker,
    title: &str,
    width: u32,
    height: u32,
) -> Result<Retained<NSWindow>, String> {
    let content_rect = NSRect::new(
        NSPoint::new(0.0, 0.0),
        NSSize::new(width as f64, height as f64),
    );
    let style =
        NSWindowStyleMask::Closable | NSWindowStyleMask::Resizable | NSWindowStyleMask::Titled;
    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            NSWindow::alloc(mtm),
            content_rect,
            style,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    window.setTitle(&ns_str(title));
    window.center();
    // Prevent AppKit from releasing the window when it is closed. The default
    // is YES for alloc/init-created windows, which causes AppKit to release
    // (and possibly deallocate) the window on close while Rust's Retained<NSWindow>
    // still holds a reference, leading to EXC_BAD_ACCESS in objc_release.
    unsafe { window.setReleasedWhenClosed(false) };
    // Receive mouse-moved events even when the cursor is outside the window
    // area (necessary when CGAssociateMouseAndMouseCursorPosition decouples
    // cursor position from hardware movement).
    window.setAcceptsMouseMovedEvents(true);
    Ok(window)
}
