// src/metal/window_delegate.rs
//
// NSWindowDelegate that tracks native-fullscreen state authoritatively.
//
// macOS native fullscreen is an animated, asynchronous transition: the
// NSWindow `FullScreen` style-mask bit lags it, so reading the bit right after
// issuing `toggleFullScreen:` (or stepping the settings menu's Window Mode row
// faster than the ~1s animation) can momentarily report the wrong state and
// toggle in the wrong direction. This delegate observes the will / did enter /
// exit fullscreen notifications and keeps a shared flag in sync, which
// `set_window_mode` / `set_window_size` read instead of the lagging style mask.
// It also captures OS-driven transitions (the green traffic-light button,
// Mission Control) that never go through the settings menu.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use objc2::rc::Retained;
use objc2::runtime::{NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{NSWindow, NSWindowDelegate, NSWindowStyleMask};
use objc2_foundation::NSNotification;

pub(crate) struct FullscreenIvars {
    is_fullscreen: Arc<AtomicBool>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "ConcinnityWindowDelegate"]
    #[ivars = FullscreenIvars]
    pub(crate) struct WindowDelegate;

    unsafe impl NSObjectProtocol for WindowDelegate {}

    unsafe impl NSWindowDelegate for WindowDelegate {
        // Fired at the start of the enter-fullscreen animation, so the flag is
        // correct as soon as a transition begins rather than at its end.
        #[unsafe(method(windowWillEnterFullScreen:))]
        fn window_will_enter_full_screen(&self, _notification: &NSNotification) {
            self.ivars().is_fullscreen.store(true, Ordering::Relaxed);
        }
        // Fired at the start of the exit-fullscreen animation.
        #[unsafe(method(windowWillExitFullScreen:))]
        fn window_will_exit_full_screen(&self, _notification: &NSNotification) {
            self.ivars().is_fullscreen.store(false, Ordering::Relaxed);
        }
        // Re-affirm at the end of each transition in case a will-callback was
        // never delivered (e.g. a transition the system cancelled and reversed).
        #[unsafe(method(windowDidEnterFullScreen:))]
        fn window_did_enter_full_screen(&self, _notification: &NSNotification) {
            self.ivars().is_fullscreen.store(true, Ordering::Relaxed);
        }
        #[unsafe(method(windowDidExitFullScreen:))]
        fn window_did_exit_full_screen(&self, _notification: &NSNotification) {
            self.ivars().is_fullscreen.store(false, Ordering::Relaxed);
        }
    }
);

impl WindowDelegate {
    fn new(mtm: objc2::MainThreadMarker, is_fullscreen: Arc<AtomicBool>) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(FullscreenIvars { is_fullscreen });
        unsafe { msg_send![super(this), init] }
    }
}

// Create a fullscreen-tracking delegate, attach it to `window`, and return the
// delegate (which the caller must keep alive: NSWindow holds its delegate as a
// zeroing weak reference) plus the shared flag both it and the renderer read.
// The flag is seeded from the window's current style mask; a freshly created
// window is not fullscreen, so this is normally false.
pub(in crate::metal) fn attach_fullscreen_delegate(
    mtm: objc2::MainThreadMarker,
    window: &NSWindow,
) -> (Retained<WindowDelegate>, Arc<AtomicBool>) {
    let is_fullscreen = Arc::new(AtomicBool::new(
        window.styleMask().contains(NSWindowStyleMask::FullScreen),
    ));
    let delegate = WindowDelegate::new(mtm, Arc::clone(&is_fullscreen));
    window.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    (delegate, is_fullscreen)
}
