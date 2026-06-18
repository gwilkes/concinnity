// src/directx/draw/text_upload.rs
//
// Persistent per-frame-slot upload buffers for the HUD text geometry. The
// composite pass uploads a small vertex + index buffer per text label every
// frame. Allocating those with `CreateCommittedResource` per label per frame is
// the classic D3D12 hot-path anti-pattern (each commit is hundreds of micros);
// for the bistro HUD that was the single largest slice of per-frame CPU.
//
// Instead each frame-in-flight slot keeps one persistently-mapped upload buffer.
// Every frame the slot's cursor is reset to zero and each label's geometry is
// appended at a rolling, aligned offset; the draw binds a sub-view into the
// shared buffer. The buffer is grown (reallocated larger) only when a frame's
// text exceeds the current capacity, which after warm-up never happens. The
// frame fence (waited before a slot is reused) guarantees the GPU has finished
// reading a slot's buffer before the CPU overwrites or grows it.

use std::cell::RefCell;

use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::texture::create_buffer;
use crate::gfx::render_types::{TextDrawCall, TextVertex};

// Sub-allocation alignment. 16 bytes satisfies the index-buffer address
// requirement (a multiple of the R16 element size) and keeps each vertex
// sub-view comfortably aligned.
const TEXT_UPLOAD_ALIGN: u64 = 16;

// First-allocation capacity for a slot's buffer. A HUD's worth of text is a few
// kilobytes, so this avoids any growth in practice while staying tiny.
const TEXT_UPLOAD_MIN_CAPACITY: u64 = 64 * 1024;

// Round `offset` up to the next multiple of `align` (a power of two).
fn align_up(offset: u64, align: u64) -> u64 {
    (offset + align - 1) & !(align - 1)
}

// New capacity for a slot that must hold at least `needed` bytes, given its
// current `capacity`. Grows geometrically (doubling from the minimum) so a burst
// of small growths amortizes, but never returns less than `needed`.
fn grow_capacity(capacity: u64, needed: u64) -> u64 {
    let mut cap = capacity.max(TEXT_UPLOAD_MIN_CAPACITY);
    while cap < needed {
        cap *= 2;
    }
    cap
}

// Total bytes a frame's text geometry consumes once each label's vertex and
// index blocks are aligned. Because every sub-allocation aligns its start up to
// `TEXT_UPLOAD_ALIGN` and a prior aligned start plus an aligned size stays
// aligned, this sum is an exact upper bound on the slot cursor after all the
// pushes, so reserving it guarantees `push` never overflows mid-frame.
pub(super) fn text_calls_byte_size(text_calls: &[TextDrawCall]) -> u64 {
    text_calls
        .iter()
        .map(|c| {
            let v = (c.vertices.len() * std::mem::size_of::<TextVertex>()) as u64;
            let i = (c.indices.len() * std::mem::size_of::<u16>()) as u64;
            align_up(v, TEXT_UPLOAD_ALIGN) + align_up(i, TEXT_UPLOAD_ALIGN)
        })
        .sum()
}

// One frame slot's persistently-mapped upload buffer. `base` is the CPU map
// pointer (null until the first allocation) and `gpu_va` its GPU virtual
// address; both are re-read whenever the buffer is grown.
struct Slot {
    buffer: Option<ID3D12Resource>,
    base: *mut u8,
    gpu_va: u64,
    capacity: u64,
    cursor: u64,
}

impl Slot {
    fn empty() -> Self {
        Slot {
            buffer: None,
            base: std::ptr::null_mut(),
            gpu_va: 0,
            capacity: 0,
            cursor: 0,
        }
    }
}

// A persistently-mapped upload buffer per frame-in-flight slot for transient
// text geometry. Interior-mutable (the composite encoders run through `&self`),
// matching the rest of the per-frame DX state.
pub(in crate::directx) struct TextUploadRing {
    slots: Vec<RefCell<Slot>>,
}

impl TextUploadRing {
    pub(in crate::directx) fn new(frames: usize) -> Self {
        TextUploadRing {
            slots: (0..frames).map(|_| RefCell::new(Slot::empty())).collect(),
        }
    }

    // Begin a frame for `frame`'s slot: reset the write cursor and ensure the
    // buffer holds at least `needed` bytes, growing (and remapping) it if not.
    // The caller must invoke this once per frame before any `push`, after the
    // frame fence has confirmed the GPU is done with this slot.
    pub(in crate::directx) fn reserve(
        &self,
        device: &ID3D12Device,
        frame: usize,
        needed: u64,
    ) -> Result<(), String> {
        let mut slot = self.slots[frame].borrow_mut();
        slot.cursor = 0;
        if needed <= slot.capacity {
            return Ok(());
        }
        let new_cap = grow_capacity(slot.capacity, needed);
        let buffer = create_buffer(
            device,
            new_cap,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut base = std::ptr::null_mut::<std::ffi::c_void>();
        unsafe { buffer.Map(0, None, Some(&mut base)) }
            .map_err(|e| format!("text upload map: {e}"))?;
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        // Replacing `buffer` drops the old resource (and unmaps it); the frame
        // fence already proved the GPU finished reading it.
        slot.buffer = Some(buffer);
        slot.base = base as *mut u8;
        slot.gpu_va = gpu_va;
        slot.capacity = new_cap;
        Ok(())
    }

    // Append `bytes` at the slot's next aligned offset and return the GPU
    // virtual address of the copy. Errors if the running total would exceed the
    // reserved capacity, which cannot happen when `reserve` was called with
    // `text_calls_byte_size` for the same calls.
    pub(in crate::directx) fn push(&self, frame: usize, bytes: &[u8]) -> Result<u64, String> {
        let mut slot = self.slots[frame].borrow_mut();
        let offset = align_up(slot.cursor, TEXT_UPLOAD_ALIGN);
        let end = offset + bytes.len() as u64;
        if end > slot.capacity {
            return Err(format!(
                "text upload overflow: need {end} bytes, reserved {}",
                slot.capacity
            ));
        }
        // SAFETY: `base` is the persistent map of a buffer of `capacity` bytes;
        // `offset + bytes.len() <= capacity` checked above; the slot is only
        // touched on the main render thread.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                slot.base.add(offset as usize),
                bytes.len(),
            );
        }
        slot.cursor = end;
        Ok(slot.gpu_va + offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_rounds_to_multiple() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(31, 16), 32);
    }

    #[test]
    fn align_up_is_idempotent_on_aligned_offsets() {
        for n in [0u64, 16, 32, 48, 1024] {
            assert_eq!(align_up(n, 16), n);
        }
    }

    #[test]
    fn grow_capacity_starts_at_minimum() {
        assert_eq!(grow_capacity(0, 1), TEXT_UPLOAD_MIN_CAPACITY);
        assert_eq!(grow_capacity(0, 0), TEXT_UPLOAD_MIN_CAPACITY);
    }

    #[test]
    fn grow_capacity_doubles_until_it_fits() {
        let need = TEXT_UPLOAD_MIN_CAPACITY * 3 + 1;
        let cap = grow_capacity(0, need);
        assert!(cap >= need);
        assert_eq!(cap, TEXT_UPLOAD_MIN_CAPACITY * 4);
    }

    #[test]
    fn grow_capacity_never_shrinks_below_existing() {
        let cap = grow_capacity(TEXT_UPLOAD_MIN_CAPACITY * 8, 10);
        assert_eq!(cap, TEXT_UPLOAD_MIN_CAPACITY * 8);
    }

    // The aligned-block sum must be an exact upper bound on the cursor after a
    // run of pushes (an aligned start plus an aligned size stays aligned), so a
    // reserve of that size can never overflow.
    #[test]
    fn byte_size_bounds_simulated_cursor() {
        let block_sizes: [u64; 4] = [3, 16, 17, 100];
        let total: u64 = block_sizes
            .iter()
            .map(|&n| align_up(n, TEXT_UPLOAD_ALIGN))
            .sum();
        let mut cursor = 0u64;
        for &n in &block_sizes {
            let offset = align_up(cursor, TEXT_UPLOAD_ALIGN);
            cursor = offset + n;
            assert!(cursor <= total, "cursor {cursor} exceeded reserved {total}");
        }
    }
}
