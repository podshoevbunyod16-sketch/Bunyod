//! Device profiles (§8): thread caps + memory budgets. Mobile profile uses
//! fewer threads; parallelism can be disabled entirely.
//!
//! IMPORTANT (honesty): these are RUNTIME SETTINGS measured on this Linux
//! x86_64 sandbox. No Android-device measurement has been performed yet, so
//! we do NOT claim Android support is complete — only that the aarch64-linux-
//! android target compiles (see docs/ANDROID.md).

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Profile {
    Mobile,
    Desktop,
    Server,
}

pub struct ProfileSettings {
    pub name: &'static str,
    pub max_threads: usize,
    /// soft RAM budget for an in-RAM store (bytes); 0 = unbounded
    pub ram_budget_bytes: usize,
    pub parallelism: bool,
}

impl Profile {
    pub fn settings(&self) -> ProfileSettings {
        match self {
            // Mobile (Termux/Android): 1-2 threads, ~256 MB budget.
            Profile::Mobile => ProfileSettings {
                name: "mobile", max_threads: 1, ram_budget_bytes: 256 * 1024 * 1024,
                parallelism: false,
            },
            Profile::Desktop => ProfileSettings {
                name: "desktop", max_threads: 4, ram_budget_bytes: 2 * 1024 * 1024 * 1024,
                parallelism: true,
            },
            Profile::Server => ProfileSettings {
                name: "server", max_threads: 0, ram_budget_bytes: 0, parallelism: true,
            },
        }
    }

    /// Build a rayon pool honoring the profile. Returns Err if pool build
    /// fails; callers should fall back to sequential execution.
    pub fn build_pool(&self) -> Result<rayon::ThreadPool, ProfilePoolError> {
        let s = self.settings();
        if !s.parallelism {
            // single-thread pool = deterministic and mobile-friendly
            rayon::ThreadPoolBuilder::new().num_threads(1).build()
                .map_err(|e| ProfilePoolError(e.to_string()))
        } else {
            let mut b = rayon::ThreadPoolBuilder::new();
            if s.max_threads > 0 { b = b.num_threads(s.max_threads); }
            b.build().map_err(|e| ProfilePoolError(e.to_string()))
        }
    }
}

#[derive(Debug)]
pub struct ProfilePoolError(pub String);
impl std::fmt::Display for ProfilePoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pool build failed: {}", self.0)
    }
}
impl std::error::Error for ProfilePoolError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mobile_is_single_thread_sequential() {
        let s = Profile::Mobile.settings();
        assert_eq!(s.max_threads, 1);
        assert!(!s.parallelism);
    }

    #[test]
    fn pool_builds_for_all_profiles() {
        for p in [Profile::Mobile, Profile::Desktop, Profile::Server] {
            let pool = p.build_pool();
            assert!(pool.is_ok(), "{p:?}: {:?}", pool.err());
        }
    }

    #[test]
    fn store_fits_mobile_budget() {
        // 1000 facts * 16 B = 16 KB << 256 MB — mobile budget ok
        use crate::compute::PackedStore;
        let mut ps = PackedStore::new();
        for i in 0..1000 { ps.put(i, i); }
        let s = Profile::Mobile.settings();
        assert!(ps.bytes_used() < s.ram_budget_bytes);
    }
}
