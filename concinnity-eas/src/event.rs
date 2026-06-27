// concinnity-eas/src/event.rs
//
// Double-buffered event queue with per-reader cursors. Replaces the
// drained-Vec-as-event pattern, which is lossy within a frame (one reader's
// drain hides the events from every other reader, and multiple sends collapse
// to last-write-wins). Here an event stays readable for two `update` cycles, so
// any reader that runs after the writer sees it, and every reader sees every
// event exactly once.
//
// Each event carries a monotonically increasing sequence id. A reader's cursor
// stores the next id it has not yet seen; reading yields every buffered event
// at or past the cursor and advances it.

pub struct Events<E> {
    // Two frame buffers. `newest` indexes the one new events go into; the other
    // holds the previous frame's events, still readable.
    buffers: [Vec<E>; 2],
    newest: usize,
    // Id assigned to the next event sent.
    next_id: usize,
    // Sequence id of the first event in each buffer.
    starts: [usize; 2],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EventCursor {
    // Next sequence id this reader has not yet consumed.
    next: usize,
}

impl<E> Default for Events<E> {
    fn default() -> Events<E> {
        Events {
            buffers: [Vec::new(), Vec::new()],
            newest: 0,
            next_id: 0,
            starts: [0, 0],
        }
    }
}

impl<E> Events<E> {
    pub fn new() -> Events<E> {
        Events::default()
    }

    // Queue an event. It becomes visible to readers immediately and stays
    // readable until the second `update` after this one.
    pub fn send(&mut self, event: E) {
        self.buffers[self.newest].push(event);
        self.next_id += 1;
    }

    // Advance one frame: retire the older buffer and start a fresh newest one.
    // Events older than two cycles are dropped.
    pub fn update(&mut self) {
        let oldest = self.newest ^ 1;
        self.buffers[oldest].clear();
        self.starts[oldest] = self.next_id;
        self.newest = oldest;
    }

    // Read every buffered event the cursor has not yet seen, in send order, and
    // advance the cursor past them.
    pub fn read(&self, cursor: &mut EventCursor) -> Vec<&E> {
        let mut out = Vec::new();
        // Visit buffers oldest-first so events come back in send order.
        let mut order = [0usize, 1usize];
        if self.starts[order[0]] > self.starts[order[1]] {
            order.swap(0, 1);
        }
        for &buffer in &order {
            let start = self.starts[buffer];
            for (offset, event) in self.buffers[buffer].iter().enumerate() {
                if start + offset >= cursor.next {
                    out.push(event);
                }
            }
        }
        cursor.next = self.next_id;
        out
    }

    // Total events currently buffered across both frames.
    pub fn len(&self) -> usize {
        self.buffers[0].len() + self.buffers[1].len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffers[0].is_empty() && self.buffers[1].is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_sees_each_event_once() {
        let mut events: Events<u32> = Events::new();
        events.send(1);
        events.send(2);
        let mut cursor = EventCursor::default();
        let first: Vec<u32> = events.read(&mut cursor).into_iter().copied().collect();
        assert_eq!(first, vec![1, 2]);
        // A second read with the same cursor sees nothing new.
        assert!(events.read(&mut cursor).is_empty());
        // A newly sent event is picked up.
        events.send(3);
        let next: Vec<u32> = events.read(&mut cursor).into_iter().copied().collect();
        assert_eq!(next, vec![3]);
    }

    #[test]
    fn multiple_readers_each_see_all_events() {
        let mut events: Events<u32> = Events::new();
        events.send(10);
        events.send(20);
        let mut a = EventCursor::default();
        let mut b = EventCursor::default();
        let read_a: Vec<u32> = events.read(&mut a).into_iter().copied().collect();
        let read_b: Vec<u32> = events.read(&mut b).into_iter().copied().collect();
        assert_eq!(read_a, vec![10, 20]);
        assert_eq!(read_b, vec![10, 20]);
    }

    #[test]
    fn events_survive_one_update_then_drop() {
        let mut events: Events<u32> = Events::new();
        events.send(1);
        events.update();
        // Still readable one cycle later, in send order with a later event.
        events.send(2);
        let mut cursor = EventCursor::default();
        let seen: Vec<u32> = events.read(&mut cursor).into_iter().copied().collect();
        assert_eq!(seen, vec![1, 2]);
        // Two updates retire the first event entirely.
        events.update();
        events.update();
        assert!(events.is_empty());
    }
}
