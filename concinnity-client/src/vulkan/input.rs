// Input snapshot returned by VkContext::take_input() each frame.
//
// The previously-duplicated InputState struct collapsed into the shared
// crate::gfx::input::RenderInput; this module re-exports it under the
// historical name so the rest of the Vulkan backend keeps compiling.

pub use crate::gfx::input::RenderInput as InputState;
