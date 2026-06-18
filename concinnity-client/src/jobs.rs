// src/jobs.rs
//
// Backend-agnostic job pool for parallelising expensive per-frame CPU work.
//
// Systems run serially in the frame loop, each holding `&mut PipelineContext`.
// This pool does not change that: it lets a single system fan its own
// data-parallel work (per-skeleton pose sampling, particle update, ...) across
// worker threads and join before `step` returns. It is not a way to run whole
// systems concurrently.
//
// The pool wraps a dedicated `rayon::ThreadPool` rather than rayon's global
// pool so the worker count and thread names are controlled. It is process-wide
// and lazily built on first use via `pool()`.

use std::sync::OnceLock;

use rayon::prelude::*;

// A dedicated thread pool for per-frame data-parallel work.
pub struct JobPool {
    pool: rayon::ThreadPool,
}

impl JobPool {
    // Build the pool. One worker per logical core is left for the main
    // thread, so the count is `available_parallelism() - 1`, clamped to at
    // least one.
    fn build() -> JobPool {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("cn-job-{i}"))
            .build()
            .expect("failed to build job thread pool");
        tracing::info!("JobPool: {threads} worker thread(s)");
        JobPool { pool }
    }

    // Apply `f` to every item in parallel, blocking until all are done.
    //
    // Each item must be independent: `f` runs concurrently across items in
    // no defined order. Inputs shorter than two items skip the pool and run
    // inline to avoid dispatch overhead.
    pub fn parallel_for<T, F>(&self, items: &mut [T], f: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        if items.len() < 2 {
            items.iter_mut().for_each(f);
            return;
        }
        self.pool.install(|| items.par_iter_mut().for_each(f));
    }

    // Run a closure inside this pool's scope so any nested rayon
    // `par_iter` / `par_iter_mut` calls dispatch to JobPool's bounded thread
    // count (`available_parallelism() - 1`) instead of rayon's global pool
    // (which defaults to every core and would starve the render thread when
    // invoked from a worker that is itself competing for CPU).
    //
    // Used by the DirectX / Metal parallel command-buffer recording; the Vulkan
    // backend records single-threaded, so it is unused under `backend_vk`.
    #[allow(dead_code)]
    pub fn install<R, F>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(f)
    }
}

// The process-wide job pool, built on first access.
pub fn pool() -> &'static JobPool {
    static POOL: OnceLock<JobPool> = OnceLock::new();
    POOL.get_or_init(JobPool::build)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_is_a_singleton() {
        assert!(std::ptr::eq(pool(), pool()));
    }

    #[test]
    fn parallel_for_visits_every_item() {
        let mut data: Vec<u32> = (0..10_000).collect();
        pool().parallel_for(&mut data, |x| *x += 1);
        assert!(data.iter().enumerate().all(|(i, &x)| x == i as u32 + 1));
    }

    #[test]
    fn parallel_for_handles_empty_and_single() {
        let mut empty: Vec<u32> = Vec::new();
        pool().parallel_for(&mut empty, |x| *x += 1);
        assert!(empty.is_empty());

        let mut single = vec![41u32];
        pool().parallel_for(&mut single, |x| *x += 1);
        assert_eq!(single, vec![42]);
    }
}
