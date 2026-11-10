//! Automatic expert tiering across media, learned from use, applied while the server runs and kept across restarts.
//!
//! Tiers, fastest first: GPU stores, the RAM arena, the page cache (file pages kept in RAM), an SSD buffer (when the
//! model file sits on a slow medium) and the model file itself. Nothing here changes which experts the router picks;
//! it changes where their bytes are and which device computes them.
//!
//! * **Usage.** The CPU worker counts every decode pick (`note_picks`), GPU-resident ones included, and which tier
//!   served it. A background thread folds the counts into a long-term score (`HALF_LIFE_TOKENS`). Usage does not
//!   depend on the hardware: one score serves every GPU combination the model is started with.
//! * **Run time.** The thread plans moves of hot experts that live on no GPU onto GPU stores that keep fixed sets
//!   otherwise (stores that already swap by recent use are left to that), each against a clearly colder resident of
//!   the same type. It copies the expert into a pinned prefetch slot; the engine starts at most one copy per token
//!   (`GpuExperts::begin_swap`: the victim leaves the store's table first and is computed by the CPU from then on,
//!   the copy lands in its slot on the side stream, the new expert becomes resident only after the copy completed).
//!   No arena copy of the victim is needed: the CPU reads it from the arena, the SSD buffer or the file.
//! * **Idle.** With no decode for `IDLE_SECS`: the page cache is warmed with the hottest experts the CPU reads from
//!   the file, and on a slow medium they are copied into an SSD buffer the CPU reads from at once.
//! * **Measurement.** Per GPU (by UUID): host link, sampled expert time per pick; per medium: read speed; per GPU
//!   combination: decode tokens/s and the share of expert reads that still hit the file.
//! * **Startup.** The learned score is blended into `--rank` before placement, and with measured GPU speeds the
//!   hottest experts go preferably to the faster GPU.
//!
//! State: `<model>.tier.json` next to the model (one per model file and overlay), else in `$DSX_STATE_DIR`,
//! `$XDG_STATE_HOME/dsx`, `~/.local/state/dsx`, `/var/tmp/dsx`. Previous version kept as `.json.prev`; devices unseen
//! for 90 days and all but the 16 most recent GPU combinations are dropped. `--tier-reset` starts from scratch. Any
//! failure only disables the affected part; inference never depends on it.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub const HALF_LIFE_TOKENS: f64 = 200_000.0;
pub const ALPHA_FULL_TOKENS: f64 = 50_000.0;
pub const ALPHA_MAX: f64 = 0.8;
pub const IDLE_SECS: u64 = 30;
const SAVE_EVERY_SECS: u64 = 300;
const FOLD_EVERY_SECS: u64 = 10;
const WARM_MAX_BYTES: u64 = 40 << 30;
const MEM_RESERVE_BYTES: u64 = 16 << 30;
const CGROUP_RESERVE_BYTES: u64 = 4 << 30;
const SSD_MIN_FREE_BYTES: u64 = 20 << 30;
const SSD_MAX_BYTES: u64 = 512 << 30;
const STEP_BYTES: u64 = 512 << 20;
/// below this measured read speed the model file's medium counts as slow (HDD, network, SATA behind NVMe)
pub const SLOW_READ_MB_S: f64 = 1000.0;
/// a planned move needs this much learned use, and a victim at most `MOVE_RATIO` of it
const MOVE_MIN_SCORE: f64 = 8.0;
const MOVE_RATIO: f64 = 0.5;
pub const PREFETCH_SLOTS: usize = 4;
pub const MAX_STORES: usize = 9;
/// one sample (expert kernels and the main GPU's waits, one layer, rotating) per this many decode tokens
pub const SAMPLE_EVERY: u64 = 64;
/// a measured wait counts as real above this (µs per layer); below it the device was not on the critical path
const WAIT_US: f64 = 20.0;
/// smoothing of the measured waits (weight of a new sample)
const WAIT_EMA: f64 = 0.05;
/// samples before the waits steer moves
const WAIT_MIN_SAMPLES: u64 = 16;
const DEVICE_KEEP_SECS: u64 = 90 * 86400;
const MAX_CONFIGS: usize = 16;
const STATE_VERSION: u64 = 2;

/// Where an expert's bytes are computed from: a GPU store (`LOC_GPU + store`), else the arena, else the file side
/// (SSD buffer, page cache, file).
pub const LOC_FILE: u8 = 0;
pub const LOC_ARENA: u8 = 1;
pub const LOC_GPU: u8 = 2;

/// Which tier served an expert's bytes for the CPU (GPU-resident picks are counted separately).
#[derive(Clone, Copy)]
pub enum Hit {
    Gpu = 0,
    Arena = 1,
    Ssd = 2,
    File = 3,
}

/// A GPU store as the engine built it: store 0 is the main GPU's.
#[derive(Clone, Debug)]
pub struct StoreInfo {
    pub uuid: String,
    pub name: String,
    pub link_gbs: f64,
    /// fixed set otherwise: tiering may move experts into it at run time
    pub tier_swaps: bool,
}

pub struct Tier {
    pub nl: usize,
    pub ne: usize,
    /// monotonic decode-pick counts per [layer * E + expert] (the CPU worker is the only writer)
    picks: Vec<AtomicU32>,
    hits: [AtomicU64; 4],
    /// per store: decode picks it computed
    store_picks: [AtomicU64; MAX_STORES],
    /// per store: sampled expert-kernel ms of one layer, and the number of samples
    gpu_ms: Mutex<Vec<(f64, u64)>>,
    /// the main GPU's waits per layer after its own work (smoothed, µs): for the CPU's experts, then for the aux GPUs'
    waits: Mutex<Waits>,
    last_decode: AtomicU64,
    /// decode passes: all of them, and those right after another (timed: `busy_us`)
    all_passes: AtomicU64,
    passes: AtomicU64,
    busy_us: AtomicU64,
    last_pass_us: AtomicU64,
    loc: Vec<AtomicU8>,
    /// per expert: its (gate/up, down) quant types folded into one key; experts swap only with the same key
    key: OnceLock<Vec<u32>>,
    stores: OnceLock<Vec<StoreInfo>>,
    prefetch: OnceLock<Prefetch>,
    ssd: OnceLock<SsdBuffer>,
    stop: AtomicBool,
    /// taken by the first model that sets tiering up in this process
    claimed: AtomicBool,
    t0: std::time::Instant,
}

static GLOBAL: OnceLock<&'static Tier> = OnceLock::new();
static RESET: AtomicBool = AtomicBool::new(false);

/// The process-wide tier state (created by `init`), or None when tiering is off for this run.
pub fn global() -> Option<&'static Tier> {
    GLOBAL.get().copied()
}

/// `--tier-reset`: forget this model's learned state and SSD buffer at the next load.
pub fn request_reset() {
    RESET.store(true, Ordering::Relaxed);
}

pub fn reset_requested() -> bool {
    RESET.load(Ordering::Relaxed)
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Tier {
    pub fn new(nl: usize, ne: usize) -> Tier {
        Tier {
            nl,
            ne,
            picks: (0..nl * ne).map(|_| AtomicU32::new(0)).collect(),
            hits: Default::default(),
            store_picks: Default::default(),
            gpu_ms: Mutex::new(vec![(0.0, 0); MAX_STORES]),
            waits: Mutex::new(Waits::default()),
            last_decode: AtomicU64::new(0),
            all_passes: AtomicU64::new(0),
            passes: AtomicU64::new(0),
            busy_us: AtomicU64::new(0),
            last_pass_us: AtomicU64::new(0),
            loc: (0..nl * ne).map(|_| AtomicU8::new(LOC_FILE)).collect(),
            key: OnceLock::new(),
            stores: OnceLock::new(),
            prefetch: OnceLock::new(),
            ssd: OnceLock::new(),
            stop: AtomicBool::new(false),
            claimed: AtomicBool::new(false),
            t0: std::time::Instant::now(),
        }
    }

    /// Register as the process-wide instance (first call wins).
    pub fn init(nl: usize, ne: usize) -> &'static Tier {
        GLOBAL.get_or_init(|| Box::leak(Box::new(Tier::new(nl, ne))))
    }

    /// True for the first caller only: tiering state belongs to one model per process.
    pub fn claim(&self, nl: usize, ne: usize) -> bool {
        self.nl == nl && self.ne == ne && !self.claimed.swap(true, Ordering::AcqRel)
    }

    /// A decode pass's picks for one layer (all experts, GPU-resident ones included).
    #[inline]
    pub fn note_picks(&self, l: usize, sel: &[i32]) {
        let base = l * self.ne;
        for &e in sel {
            if let Some(c) = self.picks.get(base + e as usize) {
                c.store(c.load(Ordering::Relaxed).wrapping_add(1), Ordering::Relaxed);
            }
        }
    }

    #[inline]
    pub fn note_hit(&self, h: Hit, n: u64) {
        self.hits[h as usize].fetch_add(n, Ordering::Relaxed);
    }

    /// A pick a GPU computed: counted for the GPU tier and for the store that holds it.
    #[inline]
    pub fn note_gpu(&self, l: usize, e: usize) {
        self.hits[Hit::Gpu as usize].fetch_add(1, Ordering::Relaxed);
        let s = self.loc(l * self.ne + e);
        if s >= LOC_GPU && ((s - LOC_GPU) as usize) < MAX_STORES {
            self.store_picks[(s - LOC_GPU) as usize].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Called once per decode pass: marks the server busy (no warming or copying meanwhile) and times the passes.
    #[inline]
    pub fn note_decode(&self) {
        self.last_decode.store(now_secs(), Ordering::Relaxed);
        self.all_passes.fetch_add(1, Ordering::Relaxed);
        let now = self.t0.elapsed().as_micros() as u64;
        let prev = self.last_pass_us.swap(now, Ordering::Relaxed);
        if prev > 0 && now - prev < 2_000_000 {
            self.busy_us.fetch_add(now - prev, Ordering::Relaxed);
            self.passes.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// One sampled layer's expert-kernel time on `store`.
    pub fn note_gpu_ms(&self, store: usize, ms: f64) {
        if store < MAX_STORES && ms.is_finite() && ms > 0.0 {
            let mut g = self.gpu_ms.lock().unwrap();
            g[store].0 += ms;
            g[store].1 += 1;
        }
    }

    /// One sampled layer: how long the main GPU waited for the CPU's experts, then for the aux GPUs' (ms).
    pub fn note_waits(&self, cpu_ms: f64, aux_ms: f64) {
        if cpu_ms.is_finite() && aux_ms.is_finite() && cpu_ms >= 0.0 && aux_ms >= 0.0 {
            self.waits.lock().unwrap().add(cpu_ms * 1000.0, aux_ms * 1000.0);
        }
    }

    pub fn waits(&self) -> Waits {
        *self.waits.lock().unwrap()
    }

    pub fn idle_for(&self) -> u64 {
        now_secs().saturating_sub(self.last_decode.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn loc(&self, i: usize) -> u8 {
        self.loc.get(i).map_or(LOC_FILE, |x| x.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn set_loc(&self, l: usize, e: usize, v: u8) {
        if let Some(x) = self.loc.get(l * self.ne + e) {
            x.store(v, Ordering::Relaxed);
        }
    }

    pub fn set_keys(&self, keys: Vec<u32>) {
        let _ = self.key.set(keys);
    }

    pub fn set_stores(&self, s: Vec<StoreInfo>) {
        let _ = self.stores.set(s);
    }

    pub fn set_prefetch(&self, p: Prefetch) {
        let _ = self.prefetch.set(p);
    }

    pub fn prefetch(&self) -> Option<&Prefetch> {
        self.prefetch.get()
    }

    /// The SSD buffer's copy of an expert ([gate, up, down]), if it holds one.
    #[inline]
    pub fn ssd_get(&self, l: usize, e: usize) -> Option<[&'static [u8]; 3]> {
        self.ssd.get().and_then(|s| s.get(l * self.ne + e))
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    fn snapshot_picks(&self) -> Vec<u32> {
        self.picks.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    fn hits(&self) -> [u64; 4] {
        [0, 1, 2, 3].map(|i| self.hits[i].load(Ordering::Relaxed))
    }
}

// ------------------------------------------------------------------------------------------------ run-time moves

/// A planned move: `new` (on no GPU) replaces `victim` in GPU store `store`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Move {
    pub new: (usize, usize),
    pub store: usize,
    pub victim: (usize, usize),
}

const SLOT_FREE: u8 = 0;
const SLOT_FILLING: u8 = 1;
const SLOT_READY: u8 = 2;
const SLOT_BUSY: u8 = 3;

/// Pinned slots the background thread fills with the next experts to move; the engine copies from them to a GPU and
/// releases a slot once that copy completed. At most `PREFETCH_SLOTS` experts are in transit.
pub struct Prefetch {
    base: usize,
    cap: usize,
    state: Vec<AtomicU8>,
    mv: Vec<Mutex<Option<(Move, [u32; 3])>>>,
}

impl Prefetch {
    /// `base`: page-locked host memory of `n * cap` bytes that outlives the process's use of it.
    pub fn new(base: *mut u8, cap: usize, n: usize) -> Prefetch {
        Prefetch { base: base as usize, cap, state: (0..n).map(|_| AtomicU8::new(SLOT_FREE)).collect(), mv: (0..n).map(|_| Mutex::new(None)).collect() }
    }

    fn claim(&self) -> Option<usize> {
        (0..self.state.len()).find(|&i| self.state[i].compare_exchange(SLOT_FREE, SLOT_FILLING, Ordering::AcqRel, Ordering::Relaxed).is_ok())
    }

    /// Copy the expert into a claimed slot and publish the move.
    fn fill(&self, i: usize, mv: Move, parts: [&[u8]; 3]) -> bool {
        let lens = parts.map(|p| p.len() as u32);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        if total > self.cap {
            self.state[i].store(SLOT_FREE, Ordering::Release);
            return false;
        }
        let dst = unsafe { std::slice::from_raw_parts_mut((self.base + i * self.cap) as *mut u8, total) };
        let mut o = 0;
        for p in parts {
            dst[o..o + p.len()].copy_from_slice(p);
            o += p.len();
        }
        *self.mv[i].lock().unwrap() = Some((mv, lens));
        self.state[i].store(SLOT_READY, Ordering::Release);
        true
    }

    fn unclaim(&self, i: usize) {
        self.state[i].store(SLOT_FREE, Ordering::Release);
    }

    /// The engine takes the next ready move (the slot stays in use until `release`).
    pub fn take(&self) -> Option<(usize, Move)> {
        for i in 0..self.state.len() {
            if self.state[i].compare_exchange(SLOT_READY, SLOT_BUSY, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                if let Some((mv, _)) = *self.mv[i].lock().unwrap() {
                    return Some((i, mv));
                }
                self.release(i);
            }
        }
        None
    }

    /// The expert bytes of slot `i` ([gate, up, down]).
    pub fn parts(&self, i: usize) -> Option<[&[u8]; 3]> {
        let (_, l) = (*self.mv[i].lock().unwrap())?;
        let p = (self.base + i * self.cap) as *const u8;
        let (a, b, c) = (l[0] as usize, l[1] as usize, l[2] as usize);
        unsafe { Some([std::slice::from_raw_parts(p, a), std::slice::from_raw_parts(p.add(a), b), std::slice::from_raw_parts(p.add(a + b), c)]) }
    }

    pub fn release(&self, i: usize) {
        *self.mv[i].lock().unwrap() = None;
        self.state[i].store(SLOT_FREE, Ordering::Release);
    }

    /// Moves planned or in transit (both experts of each are off limits for new plans).
    fn in_transit(&self) -> Vec<Move> {
        self.mv.iter().filter_map(|m| m.lock().unwrap().map(|x| x.0)).collect()
    }
}

/// Plan up to `room` moves: hottest experts on no GPU against the coldest same-key residents of stores that take
/// run-time moves, a victim at most `MOVE_RATIO` of the candidate's score. Per expert (`layer * ne + expert`):
/// `score`, `loc`, `key`; `open` per store (see `Waits::open`); moves in `busy` (planned or in transit) block both of
/// their experts.
pub fn plan_moves(ne: usize, score: &[f64], loc: &[u8], key: &[u32], stores: &[StoreInfo], open: &[bool], busy: &[Move], room: usize) -> Vec<Move> {
    let ne = ne.max(1);
    let mut off: std::collections::HashSet<usize> = busy.iter().flat_map(|m| [m.new, m.victim]).map(|(l, e)| l * ne + e).collect();
    let mut victims: HashMap<(usize, u32), Vec<(f64, usize)>> = HashMap::new();
    for i in 0..score.len() {
        let s = loc[i];
        if s < LOC_GPU {
            continue;
        }
        let st = (s - LOC_GPU) as usize;
        if !stores.get(st).is_some_and(|x| x.tier_swaps) || !open.get(st).copied().unwrap_or(true) {
            continue;
        }
        victims.entry((st, key[i])).or_default().push((score[i], i));
    }
    for v in victims.values_mut() {
        v.sort_by(|a, b| b.0.total_cmp(&a.0)); // coldest last
    }
    let mut cand: Vec<(f64, usize)> = (0..score.len()).filter(|&i| loc[i] < LOC_GPU && score[i] >= MOVE_MIN_SCORE).map(|i| (score[i], i)).collect();
    cand.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out = vec![];
    for (sc, i) in cand {
        if out.len() >= room {
            break;
        }
        if off.contains(&i) {
            continue;
        }
        let k = key[i];
        let best = victims
            .iter_mut()
            .filter(|((_, kk), _)| *kk == k)
            .filter_map(|((st, _), list)| {
                while list.last().is_some_and(|x| off.contains(&x.1)) {
                    list.pop();
                }
                list.last().map(|&(vs, vi)| (vs, vi, *st))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let Some((vs, vi, st)) = best else { continue };
        if vs > sc * MOVE_RATIO {
            continue;
        }
        victims.get_mut(&(st, k)).unwrap().pop();
        off.insert(i);
        off.insert(vi);
        out.push(Move { new: (i / ne, i % ne), store: st, victim: (vi / ne, vi % ne) });
    }
    out
}

/// Where a layer's time goes once the main GPU has done its own part: waiting for the CPU's experts, then for the
/// aux GPUs'. A device that makes the main GPU wait is on the critical path; one that does not has room.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Waits {
    pub cpu_us: f64,
    pub aux_us: f64,
    pub samples: u64,
}

impl Waits {
    fn add(&mut self, cpu_us: f64, aux_us: f64) {
        if self.samples == 0 {
            (self.cpu_us, self.aux_us) = (cpu_us, aux_us);
        } else {
            self.cpu_us += (cpu_us - self.cpu_us) * WAIT_EMA;
            self.aux_us += (aux_us - self.aux_us) * WAIT_EMA;
        }
        self.samples += 1;
    }

    /// Which stores may take more experts from the CPU (store 0 = main GPU, then the aux GPUs). Moving a hot expert from
    /// the CPU onto a GPU only shortens a layer while the CPU is what the main GPU waits for, and only onto a GPU that is
    /// not itself the last to finish: the main GPU while the CPU makes it wait, an aux GPU while it finishes before the
    /// CPU. Before enough samples every store is open.
    pub fn open(&self, stores: usize) -> Vec<bool> {
        if self.samples < WAIT_MIN_SAMPLES {
            return vec![true; stores];
        }
        let cpu_slow = self.cpu_us > WAIT_US;
        let aux_free = self.aux_us <= WAIT_US;
        (0..stores).map(|i| cpu_slow && (i == 0 || aux_free)).collect()
    }
}

// ------------------------------------------------------------------------------------------------ learned state

/// Long-term usage: decayed picks per expert and the number of decode tokens it was learned from.
#[derive(Clone, Debug, PartialEq)]
pub struct Learned {
    pub score: Vec<f64>,
    pub tokens: f64,
}

impl Learned {
    pub fn new(n: usize) -> Learned {
        Learned { score: vec![0.0; n], tokens: 0.0 }
    }

    /// Add new picks (delta of the monotonic counters) after decaying by the tokens they represent.
    pub fn fold(&mut self, delta: &[u32], top_k: usize, nl: usize) {
        let total: u64 = delta.iter().map(|&d| d as u64).sum();
        if total == 0 {
            return;
        }
        let tokens = total as f64 / (top_k.max(1) * nl.max(1)) as f64;
        let f = 0.5f64.powf(tokens / HALF_LIFE_TOKENS);
        for (s, &d) in self.score.iter_mut().zip(delta) {
            *s = *s * f + d as f64;
        }
        self.tokens += tokens;
    }

    pub fn alpha(&self) -> f64 {
        (self.tokens / ALPHA_FULL_TOKENS * ALPHA_MAX).min(ALPHA_MAX)
    }
}

/// Everything the state file keeps: usage (hardware-independent), device measurements, per-combination figures.
#[derive(Clone, Debug, PartialEq)]
pub struct Saved {
    pub learned: Learned,
    pub devices: serde_json::Map<String, serde_json::Value>,
    pub configs: serde_json::Map<String, serde_json::Value>,
}

impl Saved {
    pub fn new(n: usize) -> Saved {
        Saved { learned: Learned::new(n), devices: Default::default(), configs: Default::default() }
    }

    /// A GPU's measured cost per expert pick (µs), when known.
    pub fn expert_us(&self, uuid: &str) -> Option<f64> {
        self.devices.get(uuid)?.get("expert_us")?.as_f64().filter(|v| *v > 0.0)
    }

    /// Drop devices unseen for `DEVICE_KEEP_SECS` and all but the `MAX_CONFIGS` most recent combinations.
    pub fn rotate(&mut self, now: u64) {
        self.devices.retain(|_, v| v["seen"].as_u64().is_some_and(|s| now.saturating_sub(s) <= DEVICE_KEEP_SECS));
        if self.configs.len() > MAX_CONFIGS {
            let mut seen: Vec<(u64, String)> = self.configs.iter().map(|(k, v)| (v["seen"].as_u64().unwrap_or(0), k.clone())).collect();
            seen.sort();
            for (_, k) in seen.iter().take(self.configs.len() - MAX_CONFIGS) {
                self.configs.remove(k);
            }
        }
    }
}

/// Identity of one file: canonical path, size, modification time.
pub fn model_id(path: &Path) -> String {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    match std::fs::metadata(path) {
        Ok(m) => format!("{}|{}|{}", canon.display(), m.len(), m.mtime()),
        Err(_) => canon.display().to_string(),
    }
}

/// Identity of a model made of `paths` (shards, then any overlay): any change of any file starts a new state.
pub fn models_id(paths: &[PathBuf]) -> String {
    paths.iter().map(|p| model_id(p)).collect::<Vec<_>>().join(";")
}

fn clean(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' || c == '+' { c } else { '_' }).collect()
}

/// Shards of one split GGUF share the name before `-NNNNN-of-NNNNN`.
fn shard_stem(name: &str) -> Option<&str> {
    let i = name.find("-of-")?;
    let head = &name[..i];
    let j = head.rfind('-')?;
    head[j + 1..].chars().all(|c| c.is_ascii_digit()).then_some(&head[..j])
}

/// The state's base name: the first file's name, plus `+<stem>` for every file that is not a shard of it (overlays).
pub fn state_name(paths: &[PathBuf]) -> String {
    let fname = |p: &PathBuf| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    let Some(first) = paths.first() else { return "model".into() };
    let f0 = fname(first);
    let stem0 = shard_stem(&f0).map(str::to_owned);
    let mut name = f0.clone();
    for p in &paths[1..] {
        let f = fname(p);
        if stem0.is_some() && shard_stem(&f).map(str::to_owned) == stem0 {
            continue;
        }
        let stem = Path::new(&f).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or(f);
        name += &format!("+{stem}");
    }
    clean(&name)
}

fn state_dir() -> PathBuf {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let cands: Vec<PathBuf> = [
        env("DSX_STATE_DIR"),
        env("XDG_STATE_HOME").map(|p| p.join("dsx")),
        env("HOME").map(|p| p.join(".local/state/dsx")),
        Some(PathBuf::from("/var/tmp/dsx")),
    ]
    .into_iter()
    .flatten()
    .collect();
    for d in &cands {
        if std::fs::create_dir_all(d).is_ok() && writable(d) {
            return d.clone();
        }
    }
    cands.last().cloned().unwrap()
}

fn writable(d: &Path) -> bool {
    let probe = d.join(format!(".dsx-probe-{}", std::process::id()));
    let ok = File::create(&probe).is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

/// `<model dir>/<name>.tier.json` when that directory is writable, else the same name in the state directory.
pub fn state_path(paths: &[PathBuf]) -> PathBuf {
    let name = format!("{}.tier.json", state_name(paths));
    match paths.first().and_then(|p| p.parent()).filter(|d| writable(d)) {
        Some(d) => d.join(name),
        None => state_dir().join(name),
    }
}

/// Where an earlier version (one model file, state directory only) kept it.
pub fn legacy_state_path(paths: &[PathBuf]) -> Option<PathBuf> {
    (paths.len() == 1).then(|| state_dir().join(format!("{}.tier.json", state_name(paths))))
}

/// Read the state if it belongs to this model and shape; None otherwise (a new model starts fresh). Reads version 1
/// (usage only) as well.
pub fn load(path: &Path, id: &str, nl: usize, ne: usize) -> Option<Saved> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    if v["layers"].as_u64()? as usize != nl || v["experts"].as_u64()? as usize != ne {
        return None;
    }
    let (model, usage) = match v["version"].as_u64()? {
        1 => (v["model"].as_str()?.to_owned(), &v),
        STATE_VERSION => (v["model"]["id"].as_str()?.to_owned(), &v["usage"]),
        _ => return None,
    };
    if model != id {
        return None;
    }
    let score: Vec<f64> = usage["score"].as_array()?.iter().map(|x| x.as_f64().unwrap_or(0.0).max(0.0)).collect();
    if score.len() != nl * ne {
        return None;
    }
    let obj = |k: &str| v.get(k).and_then(|x| x.as_object()).cloned().unwrap_or_default();
    Some(Saved { learned: Learned { score, tokens: usage["tokens"].as_f64().unwrap_or(0.0).max(0.0) }, devices: obj("devices"), configs: obj("configs") })
}

/// Write the state atomically (temp file + rename); the previous version stays as `.prev` (rotation: two files).
pub fn save(path: &Path, id: &str, files: &[String], nl: usize, ne: usize, s: &Saved, stats: serde_json::Value) -> std::io::Result<()> {
    let v = serde_json::json!({
        "version": STATE_VERSION,
        "model": {"id": id, "files": files},
        "layers": nl, "experts": ne,
        "usage": {"tokens": s.learned.tokens, "score": s.learned.score.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>()},
        "devices": s.devices, "configs": s.configs, "stats": stats, "saved": now_secs(),
    });
    // per-process temp name: two servers on the same model never write into one temp file
    let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(serde_json::to_string(&v)?.as_bytes())?;
        f.sync_data()?;
    }
    if path.exists() {
        let _ = std::fs::rename(path, path.with_extension("json.prev"));
    }
    std::fs::rename(&tmp, path)
}

/// `--tier-reset`: remove the state (and its previous version), the legacy state and the SSD buffer of this model.
pub fn reset_files(state: &Path, legacy: Option<&Path>, name: &str) -> usize {
    let mut n = 0;
    let mut rm = |p: PathBuf| {
        if std::fs::remove_file(&p).is_ok() {
            n += 1;
        }
    };
    rm(state.to_path_buf());
    rm(state.with_extension("json.prev"));
    if let Some(l) = legacy {
        rm(l.to_path_buf());
        rm(l.with_extension("json.prev"));
    }
    for d in ssd_dirs() {
        rm(d.join(format!("{name}.ssd-buffer")));
        rm(d.join(format!("{name}.ssd-buffer.json")));
    }
    n
}

/// The startup ranking: `(1 - α)` × shipped calibration counts + `α` × learned score, both normalized; experts the
/// shipped ranking does not list keep their order behind. With no learned tokens the shipped order is unchanged.
pub fn blend_rank(base: &[(usize, usize, u64)], learned: &Learned, nl: usize, ne: usize) -> Vec<(usize, usize)> {
    let a = learned.alpha();
    let mut pos: HashMap<(usize, usize), usize> = HashMap::new();
    let mut cnt = vec![0f64; nl * ne];
    for (i, &(l, e, n)) in base.iter().enumerate() {
        if l < nl && e < ne {
            pos.entry((l, e)).or_insert(i);
            cnt[l * ne + e] = n as f64;
        }
    }
    let bsum: f64 = cnt.iter().sum::<f64>().max(1e-9);
    let lsum: f64 = learned.score.iter().sum::<f64>().max(1e-9);
    let score = |i: usize| (1.0 - a) * cnt[i] / bsum + a * learned.score.get(i).copied().unwrap_or(0.0) / lsum;
    let mut keys: Vec<(usize, usize)> = (0..ne).flat_map(|e| (0..nl).map(move |l| (l, e))).collect();
    keys.sort_by(|x, y| {
        let (i, j) = (x.0 * ne + x.1, y.0 * ne + y.1);
        score(j).total_cmp(&score(i)).then(pos.get(x).unwrap_or(&usize::MAX).cmp(pos.get(y).unwrap_or(&usize::MAX)))
    });
    keys
}

/// Relative cost per expert pick of each store (fastest = 1) when every store has a measurement, for placement.
pub fn store_costs(saved: &Saved, uuids: &[String]) -> Option<Vec<f64>> {
    let us: Vec<f64> = uuids.iter().map(|u| saved.expert_us(u)).collect::<Option<_>>()?;
    let min = us.iter().copied().fold(f64::INFINITY, f64::min);
    (min > 0.0 && min.is_finite()).then(|| us.iter().map(|u| u / min).collect())
}

// ------------------------------------------------------------------------------------------------ storage facts

fn dev_parts(dev: u64) -> (u64, u64) {
    let major = ((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff);
    let minor = (dev & 0xff) | ((dev >> 12) & !0xff);
    (major, minor)
}

/// Whether the file lives on a rotational disk (HDD), from /sys; None if unknown.
pub fn rotational(path: &Path) -> Option<bool> {
    let (ma, mi) = dev_parts(std::fs::metadata(path).ok()?.dev());
    let sys = std::fs::canonicalize(format!("/sys/dev/block/{ma}:{mi}")).ok()?;
    for d in sys.ancestors() {
        if let Ok(s) = std::fs::read_to_string(d.join("queue/rotational")) {
            return Some(s.trim() == "1");
        }
    }
    None
}

/// `disk:<major>:<minor>` of the device holding `path` (the key of its measurements).
pub fn disk_key(path: &Path) -> Option<String> {
    let (ma, mi) = dev_parts(std::fs::metadata(path).ok()?.dev());
    Some(format!("disk:{ma}:{mi}"))
}

/// Read speed of the medium under `path` in MB/s: 32 reads of 2 MiB spread over the file, bypassing the page cache
/// (O_DIRECT). None when the file is small or O_DIRECT is unavailable.
pub fn measure_read_mb_s(path: &Path) -> Option<f64> {
    use std::os::unix::fs::OpenOptionsExt;
    const O_DIRECT: i32 = if cfg!(any(target_arch = "aarch64", target_arch = "arm")) { 0o200000 } else { 0o40000 };
    const CHUNK: usize = 2 << 20;
    let len = std::fs::metadata(path).ok()?.len() as usize;
    if len < 64 * CHUNK {
        return None;
    }
    let f = std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(path).ok()?;
    let layout = std::alloc::Layout::from_size_align(CHUNK, 4096).ok()?;
    let buf = unsafe { std::alloc::alloc(layout) };
    if buf.is_null() {
        return None;
    }
    let b = unsafe { std::slice::from_raw_parts_mut(buf, CHUNK) };
    let t = std::time::Instant::now();
    let mut done = 0usize;
    for k in 0..32usize {
        // spread over the file, odd multiples so repeated runs hit the same places (no cache: O_DIRECT)
        let off = ((len - CHUNK) / 33 * (k + 1)) / 4096 * 4096;
        match f.read_at(b, off as u64) {
            Ok(n) if n > 0 => done += n,
            _ => break,
        }
    }
    let s = t.elapsed().as_secs_f64();
    unsafe { std::alloc::dealloc(buf, layout) };
    (done >= 16 * CHUNK && s > 0.0).then(|| done as f64 / s / 1e6)
}

#[repr(C)]
struct StatVfs {
    bsize: u64,
    frsize: u64,
    blocks: u64,
    bfree: u64,
    bavail: u64,
    files: u64,
    ffree: u64,
    favail: u64,
    fsid: u64,
    flag: u64,
    namemax: u64,
    spare: [i32; 6],
}
unsafe extern "C" {
    fn statvfs(path: *const std::ffi::c_char, buf: *mut StatVfs) -> i32;
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
}

pub fn free_bytes(dir: &Path) -> Option<u64> {
    let c = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes()).ok()?;
    let mut s: StatVfs = unsafe { std::mem::zeroed() };
    (unsafe { statvfs(c.as_ptr(), &mut s) } == 0).then(|| s.bavail * s.frsize.max(1))
}

/// Bytes the page cache may still take: min(MemAvailable − reserve, cgroup headroom − reserve).
pub fn ram_headroom() -> u64 {
    let avail = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()))
        .map_or(0, |kb| kb * 1024);
    let mut room = avail.saturating_sub(MEM_RESERVE_BYTES);
    if let Some(cg) = std::fs::read_to_string("/proc/self/cgroup").ok().and_then(|t| t.lines().find_map(|l| l.strip_prefix("0::").map(str::to_owned))) {
        let d = Path::new("/sys/fs/cgroup").join(cg.trim_start_matches('/'));
        let rd = |n: &str| std::fs::read_to_string(d.join(n)).ok().and_then(|s| s.trim().parse::<u64>().ok());
        if let (Some(max), Some(cur)) = (rd("memory.max"), rd("memory.current")) {
            room = room.min(max.saturating_sub(cur).saturating_sub(CGROUP_RESERVE_BYTES));
        }
    }
    room
}

const MADV_WILLNEED: i32 = 3;
const MADV_DONTNEED: i32 = 4;

fn advise(b: &[u8], advice: i32) {
    if b.is_empty() {
        return;
    }
    let p = b.as_ptr() as usize;
    let start = p & !4095;
    unsafe { madvise(start as *mut _, p + b.len() - start, advice) };
}

/// Fault the bytes into the page cache (one read per 4 KiB page); returns the bytes covered.
pub fn touch(b: &[u8]) -> u64 {
    advise(b, MADV_WILLNEED);
    let mut acc = 0u8;
    for i in (0..b.len()).step_by(4096) {
        acc ^= unsafe { std::ptr::read_volatile(b.as_ptr().add(i)) };
    }
    std::hint::black_box(acc);
    b.len() as u64
}

// ------------------------------------------------------------------------------------------------ SSD buffer

/// Copies of hot experts on an SSD when the model file sits on a slow medium. One preallocated (sparse) file,
/// mapped once; entries are appended, written and synced before they become visible, and an index file names them
/// so the buffer survives restarts. Entries are never rewritten while the process runs (readers hold no locks).
pub struct SsdBuffer {
    file: File,
    map: &'static [u8],
    path: PathBuf,
    index: PathBuf,
    cap: u64,
    used: AtomicU64,
    /// per expert: offset + 1 of its [gate | up | down] copy, 0 = not held
    slot: Vec<AtomicU64>,
    lens: Vec<[u32; 3]>,
    entries: Mutex<Vec<(usize, u64)>>,
    id: String,
}

impl SsdBuffer {
    /// Open or create the buffer in `dir` for the model `id`; `lens` gives each expert's three blob sizes.
    pub fn open(dir: &Path, id: &str, name: &str, lens: Vec<[u32; 3]>, cap: u64) -> std::io::Result<SsdBuffer> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{name}.ssd-buffer"));
        let index = dir.join(format!("{name}.ssd-buffer.json"));
        let mut entries: Vec<(usize, u64)> = vec![];
        let mut used = 0u64;
        let prior = std::fs::read(&index).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        let reuse = prior.as_ref().is_some_and(|v| v["model"].as_str() == Some(id) && v["cap"].as_u64() == Some(cap))
            && std::fs::metadata(&path).is_ok_and(|m| m.len() == cap);
        if reuse {
            for x in prior.as_ref().unwrap()["entries"].as_array().into_iter().flatten() {
                let (Some(i), Some(off)) = (x[0].as_u64(), x[1].as_u64()) else { continue };
                let i = i as usize;
                let Some(len) = lens.get(i).map(|l| l.iter().map(|&n| n as u64).sum::<u64>()) else { continue };
                if off + len <= cap {
                    entries.push((i, off));
                    used = used.max(off + len);
                }
            }
        }
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(!reuse).open(&path)?;
        if !reuse {
            file.set_len(cap)?;
        }
        let map: &'static memmap2::Mmap = Box::leak(Box::new(unsafe { memmap2::MmapOptions::new().len(cap as usize).map(&file)? }));
        let slot: Vec<AtomicU64> = (0..lens.len()).map(|_| AtomicU64::new(0)).collect();
        for &(i, off) in &entries {
            slot[i].store(off + 1, Ordering::Release);
        }
        Ok(SsdBuffer { file, map: &map[..], path, index, cap, used: AtomicU64::new(used.next_multiple_of(4096)), slot, lens, entries: Mutex::new(entries), id: id.to_owned() })
    }

    pub fn get(&self, i: usize) -> Option<[&'static [u8]; 3]> {
        let off = self.slot.get(i)?.load(Ordering::Acquire);
        if off == 0 {
            return None;
        }
        let o = (off - 1) as usize;
        let [a, b, c] = self.lens[i].map(|n| n as usize);
        Some([&self.map[o..o + a], &self.map[o + a..o + a + b], &self.map[o + a + b..o + a + b + c]])
    }

    pub fn holds(&self, i: usize) -> bool {
        self.slot.get(i).is_some_and(|s| s.load(Ordering::Relaxed) != 0)
    }

    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// Append one expert's three blobs; false when the buffer is full.
    pub fn put(&self, i: usize, parts: [&[u8]; 3]) -> std::io::Result<bool> {
        if self.holds(i) {
            return Ok(true);
        }
        let len: u64 = parts.iter().map(|p| p.len() as u64).sum();
        let off = self.used();
        if off + len > self.cap {
            return Ok(false);
        }
        let mut o = off;
        for p in parts {
            self.file.write_all_at(p, o)?;
            o += p.len() as u64;
        }
        self.file.sync_data()?;
        self.used.store((off + len).next_multiple_of(4096), Ordering::Relaxed);
        self.slot[i].store(off + 1, Ordering::Release);
        self.entries.lock().unwrap().push((i, off));
        Ok(true)
    }

    /// Persist the index (atomically) so the next start finds the copies.
    pub fn save_index(&self) -> std::io::Result<()> {
        let e = self.entries.lock().unwrap().clone();
        let v = serde_json::json!({"model": self.id, "cap": self.cap, "entries": e});
        let tmp = self.index.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&v)?)?;
        std::fs::rename(tmp, &self.index)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn ssd_dirs() -> Vec<PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    [
        env("DSX_CACHE_DIR"),
        Some(state_dir()),
        env("XDG_CACHE_HOME").map(|p| p.join("dsx")),
        env("HOME").map(|p| p.join(".cache/dsx")),
        Some(PathBuf::from("/var/tmp/dsx")),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Where to put the SSD buffer: the first writable directory on another, non-rotational device with room.
pub fn ssd_dir(model: &Path) -> Option<PathBuf> {
    let model_dev = std::fs::metadata(model).ok().map(|m| m.dev());
    ssd_dirs().into_iter().find(|d| {
        std::fs::create_dir_all(d).is_ok()
            && writable(d)
            && rotational(d) == Some(false)
            && std::fs::metadata(d).ok().map(|m| m.dev()) != model_dev
            && free_bytes(d).is_some_and(|f| f > SSD_MIN_FREE_BYTES)
    })
}

// ------------------------------------------------------------------------------------------------ background work

/// What the background thread needs from the model: each expert's three byte slices (where the CPU would read them).
pub trait ExpertBytes: Sync {
    fn parts(&self, l: usize, e: usize) -> [&[u8]; 3];
}

pub struct Background {
    pub tier: &'static Tier,
    pub state: PathBuf,
    pub id: String,
    pub files: Vec<String>,
    pub top_k: usize,
    pub saved: Saved,
    pub model_disk: Option<String>,
    pub model_read_mb_s: Option<f64>,
    pub slow_media: bool,
}

#[derive(Default, Debug)]
pub struct Stats {
    pub warm_bytes: u64,
    pub warm_secs: f64,
    pub ssd_bytes: u64,
    pub ssd_secs: f64,
    pub saves: u64,
    pub moves_planned: u64,
}

impl Background {
    /// Experts the CPU reads from the file side (no GPU, no arena), hottest first.
    fn order(&self) -> Vec<(usize, usize)> {
        let ne = self.tier.ne;
        let mut v: Vec<(usize, usize)> = (0..self.tier.nl * ne).filter(|&i| self.tier.loc(i) == LOC_FILE).map(|i| (i / ne, i % ne)).collect();
        v.sort_by(|a, b| self.saved.learned.score[b.0 * ne + b.1].total_cmp(&self.saved.learned.score[a.0 * ne + a.1]));
        v
    }

    /// GPU-combination key: the stores' UUIDs, main first.
    fn config_key(&self) -> Option<String> {
        let s = self.tier.stores.get()?;
        (!s.is_empty()).then(|| s.iter().map(|x| x.uuid.clone()).collect::<Vec<_>>().join("+"))
    }

    /// Fold this run's measurements into the device and combination records.
    fn measure(&mut self) {
        let now = now_secs();
        let nl = self.tier.nl.max(1) as f64;
        let passes = self.tier.passes.load(Ordering::Relaxed);
        let all = self.tier.all_passes.load(Ordering::Relaxed);
        if let Some(stores) = self.tier.stores.get() {
            let ms = self.tier.gpu_ms.lock().unwrap().clone();
            for (i, s) in stores.iter().enumerate() {
                let picks = self.tier.store_picks[i.min(MAX_STORES - 1)].load(Ordering::Relaxed) as f64;
                let mut d = self.saved.devices.get(&s.uuid).and_then(|v| v.as_object().cloned()).unwrap_or_default();
                d.insert("name".into(), s.name.clone().into());
                d.insert("link_gbs".into(), ((s.link_gbs * 10.0).round() / 10.0).into());
                d.insert("seen".into(), now.into());
                let per_layer = if all > 0 { picks / all as f64 / nl } else { 0.0 };
                if let Some(&(sum, n)) = ms.get(i)
                    && n >= 8
                    && per_layer >= 0.05
                {
                    let us = sum / n as f64 * 1000.0 / per_layer;
                    // smooth across runs: the new run weighs like its sample count, at most 3/4
                    let w = (n as f64 / 64.0).min(0.75);
                    let old = d.get("expert_us").and_then(|v| v.as_f64()).filter(|v| *v > 0.0);
                    let v = old.map_or(us, |o| o * (1.0 - w) + us * w);
                    d.insert("expert_us".into(), ((v * 10.0).round() / 10.0).into());
                    d.insert("samples".into(), (d.get("samples").and_then(|x| x.as_u64()).unwrap_or(0) + n).into());
                }
                self.saved.devices.insert(s.uuid.clone(), serde_json::Value::Object(d));
            }
        }
        if let (Some(k), Some(mbs)) = (&self.model_disk, self.model_read_mb_s) {
            self.saved.devices.insert(k.clone(), serde_json::json!({"read_mb_s": mbs.round(), "seen": now}));
        }
        if let Some(key) = self.config_key() {
            let busy = self.tier.busy_us.load(Ordering::Relaxed) as f64 / 1e6;
            let h = self.tier.hits();
            let tot = h.iter().sum::<u64>().max(1) as f64;
            let mut c = self.saved.configs.get(&key).and_then(|v| v.as_object().cloned()).unwrap_or_default();
            if passes >= 100 && busy > 0.0 {
                c.insert("decode_tok_s".into(), ((passes as f64 / busy * 100.0).round() / 100.0).into());
            }
            c.insert("share_from_file".into(), (((h[3] as f64 / tot) * 1000.0).round() / 1000.0).into());
            let w = self.tier.waits();
            if w.samples >= WAIT_MIN_SAMPLES {
                c.insert("wait_for_cpu_us".into(), w.cpu_us.round().into());
                c.insert("wait_for_aux_us".into(), w.aux_us.round().into());
            }
            let cpu = h[1] + h[2] + h[3];
            if all > 0 {
                c.insert("cpu_picks_per_token".into(), ((cpu as f64 / all as f64 * 100.0).round() / 100.0).into());
            }
            c.insert("seen".into(), now.into());
            self.saved.configs.insert(key, serde_json::Value::Object(c));
        }
        self.saved.rotate(now);
    }

    fn stats_json(&self, st: &Stats) -> serde_json::Value {
        let h = self.tier.hits();
        let tot = (h.iter().sum::<u64>()).max(1) as f64;
        let rate = |b: u64, s: f64| if s > 0.05 && b >= 64 << 20 { Some((b as f64 / s / 1e6).round()) } else { None };
        let ne = self.tier.ne;
        let on_gpu = (0..self.tier.nl * ne).filter(|&i| self.tier.loc(i) >= LOC_GPU).count();
        serde_json::json!({
            "picks": {"gpu": h[0], "arena": h[1], "ssd": h[2], "file": h[3]},
            "share_from_file": ((h[3] as f64 / tot) * 1000.0).round() / 1000.0,
            "experts_on_gpus": on_gpu,
            "file_side_experts": (0..self.tier.nl * ne).filter(|&i| self.tier.loc(i) == LOC_FILE).count(),
            "moves_planned": st.moves_planned,
            "wait_for_cpu_us": self.tier.waits().cpu_us.round(),
            "wait_for_aux_us": self.tier.waits().aux_us.round(),
            "wait_samples": self.tier.waits().samples,
            "page_cache_warm_gb": (st.warm_bytes as f64 / 1e9 * 10.0).round() / 10.0,
            "model_read_mb_s": self.model_read_mb_s.map(f64::round).or(rate(st.warm_bytes, st.warm_secs)),
            "ssd_buffer": self.tier.ssd.get().map(|s| serde_json::json!({"path": s.path(), "used_gb": (s.used() as f64 / 1e9 * 10.0).round() / 10.0,
                                                                         "write_mb_s": rate(st.ssd_bytes, st.ssd_secs)})),
            "slow_media": self.slow_media,
            "learned_tokens": self.saved.learned.tokens.round(),
            "learned_share": (self.saved.learned.alpha() * 1000.0).round() / 1000.0,
        })
    }

    /// Plan and prefetch run-time moves into free slots. Returns the number planned.
    pub fn prefetch_moves(&self, m: &dyn ExpertBytes) -> usize {
        let (Some(pf), Some(stores), Some(key)) = (self.tier.prefetch.get(), self.tier.stores.get(), self.tier.key.get()) else { return 0 };
        if !stores.iter().any(|s| s.tier_swaps) {
            return 0;
        }
        let free = pf.state.iter().filter(|s| s.load(Ordering::Relaxed) == SLOT_FREE).count();
        if free == 0 {
            return 0;
        }
        let ne = self.tier.ne;
        let loc: Vec<u8> = (0..self.tier.nl * ne).map(|i| self.tier.loc(i)).collect();
        let busy = pf.in_transit();
        let open = self.tier.waits().open(stores.len());
        let moves = plan_moves(ne, &self.saved.learned.score, &loc, key, stores, &open, &busy, free);
        let mut n = 0;
        for mv in moves {
            let Some(i) = pf.claim() else { break };
            let (l, e) = mv.new;
            let parts = self.tier.ssd_get(l, e).unwrap_or_else(|| m.parts(l, e));
            if pf.fill(i, mv, parts) {
                n += 1;
            } else {
                pf.unclaim(i);
            }
        }
        n
    }

    /// One idle step: warm the page cache, then extend the SSD buffer. Returns bytes moved.
    pub fn idle_step(&self, m: &dyn ExpertBytes, st: &mut Stats, warmed: &mut std::collections::HashSet<(usize, usize)>) -> u64 {
        let mut moved = 0u64;
        let order = self.order();
        let budget = WARM_MAX_BYTES.saturating_sub(st.warm_bytes).min(ram_headroom());
        let t = std::time::Instant::now();
        for &(l, e) in &order {
            if moved >= STEP_BYTES.min(budget) {
                break;
            }
            if warmed.contains(&(l, e)) {
                continue;
            }
            if self.tier.idle_for() < IDLE_SECS {
                break; // a request arrived: stop at once
            }
            let p = match self.tier.ssd_get(l, e) {
                Some(s) => s,
                None => m.parts(l, e),
            };
            moved += p.iter().map(|b| touch(b)).sum::<u64>();
            warmed.insert((l, e));
        }
        if moved > 0 {
            st.warm_bytes += moved;
            st.warm_secs += t.elapsed().as_secs_f64();
        }
        if let Some(ssd) = self.tier.ssd.get() {
            let t = std::time::Instant::now();
            let mut copied = 0u64;
            for &(l, e) in &order {
                if copied >= STEP_BYTES || self.tier.idle_for() < IDLE_SECS {
                    break;
                }
                let i = l * self.tier.ne + e;
                if ssd.holds(i) {
                    continue;
                }
                let p = m.parts(l, e);
                match ssd.put(i, p) {
                    Ok(true) => {
                        copied += p.iter().map(|b| b.len() as u64).sum::<u64>();
                        for b in p {
                            advise(b, MADV_DONTNEED); // the slow medium's pages are no longer needed in RAM
                        }
                    }
                    _ => break,
                }
            }
            if copied > 0 {
                st.ssd_bytes += copied;
                st.ssd_secs += t.elapsed().as_secs_f64();
                let _ = ssd.save_index();
            }
            moved += copied;
        }
        moved
    }

    fn save_now(&mut self, st: &Stats) -> bool {
        self.measure();
        let stats = self.stats_json(st);
        save(&self.state, &self.id, &self.files, self.tier.nl, self.tier.ne, &self.saved, stats).is_ok()
    }

    /// The loop: prefetch run-time moves every second, fold usage every 10 s, work and save while idle.
    pub fn run(mut self, m: &'static dyn ExpertBytes) {
        let mut last = self.tier.snapshot_picks();
        let mut st = Stats::default();
        let mut warmed = std::collections::HashSet::new();
        let mut last_save = std::time::Instant::now();
        let mut last_fold = std::time::Instant::now();
        let mut dirty = false;
        let mut last_report = std::time::Instant::now();
        while !self.tier.stop.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_secs(1));
            if last_fold.elapsed().as_secs() >= FOLD_EVERY_SECS {
                last_fold = std::time::Instant::now();
                let now = self.tier.snapshot_picks();
                let delta: Vec<u32> = now.iter().zip(&last).map(|(a, b)| a.wrapping_sub(*b)).collect();
                last = now;
                if delta.iter().any(|&d| d > 0) {
                    self.saved.learned.fold(&delta, self.top_k, self.tier.nl);
                    dirty = true;
                }
            }
            st.moves_planned += self.prefetch_moves(m) as u64;
            if self.tier.idle_for() < IDLE_SECS {
                continue;
            }
            self.idle_step(m, &mut st, &mut warmed);
            if dirty && last_save.elapsed().as_secs() >= SAVE_EVERY_SECS {
                if self.save_now(&st) {
                    st.saves += 1;
                }
                last_save = std::time::Instant::now();
                dirty = false;
            }
            if last_report.elapsed().as_secs() >= 600 {
                eprintln!("tier: {}", self.stats_json(&st));
                last_report = std::time::Instant::now();
            }
        }
        let _ = self.save_now(&st);
    }
}

/// Set up the SSD buffer for a model on a slow medium; returns whether one is active.
pub fn attach_ssd(tier: &Tier, model: &Path, id: &str, name: &str, lens: Vec<[u32; 3]>, file_side_bytes: u64) -> bool {
    let Some(dir) = ssd_dir(model) else {
        eprintln!("tier: model on a slow medium, but no SSD directory with room on another device (DSX_CACHE_DIR)");
        return false;
    };
    let cap = (free_bytes(&dir).unwrap_or(0) / 2).min(SSD_MAX_BYTES).min(file_side_bytes.next_multiple_of(1 << 30)).next_multiple_of(4096);
    if cap < 1 << 30 {
        return false;
    }
    match SsdBuffer::open(&dir, id, name, lens, cap) {
        Ok(b) => {
            eprintln!("tier: SSD buffer {} ({:.0} GB, {:.1} GB held)", b.path().display(), cap as f64 / 1e9, b.used() as f64 / 1e9);
            tier.ssd.set(b).is_ok()
        }
        Err(e) => {
            eprintln!("tier: SSD buffer unavailable: {e}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_decays_and_counts_tokens() {
        let mut l = Learned::new(4);
        l.fold(&[6, 0, 0, 0], 6, 1);
        assert_eq!(l.tokens, 1.0);
        assert!((l.score[0] - 6.0).abs() < 1e-9);
        l.score[1] = 10.0;
        let big = (HALF_LIFE_TOKENS * 6.0) as u32;
        l.fold(&[0, 0, big, 0], 6, 1);
        assert!((l.score[1] - 5.0).abs() < 1e-6, "{}", l.score[1]);
        assert!(l.alpha() <= ALPHA_MAX + 1e-12);
    }

    #[test]
    fn blend_keeps_shipped_order_without_learning_and_follows_use_with_it() {
        let base: Vec<(usize, usize, u64)> = vec![(0, 0, 100), (1, 0, 90), (0, 1, 80), (1, 1, 70)];
        let none = Learned::new(4);
        assert_eq!(blend_rank(&base, &none, 2, 2), vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        let mut used = Learned::new(4);
        used.score[1 * 2 + 1] = 1e6;
        used.tokens = ALPHA_FULL_TOKENS;
        let r = blend_rank(&base, &used, 2, 2);
        assert_eq!(r[0], (1, 1));
        assert_eq!(r.len(), 4);
        let r = blend_rank(&base[..1], &none, 2, 2);
        let mut s = r.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 4);
        assert_eq!(r[0], (0, 0));
    }

    #[test]
    fn state_roundtrip_rotation_identity_and_version1() {
        let d = std::env::temp_dir().join(format!("dsx-tier-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("m.tier.json");
        let mut s = Saved::new(6);
        s.learned.score[3] = 2.5;
        s.learned.tokens = 7.0;
        s.devices.insert("GPU-a".into(), serde_json::json!({"name": "x", "expert_us": 40.0, "seen": now_secs()}));
        save(&p, "id-a", &["m.gguf".into()], 2, 3, &s, serde_json::json!({})).unwrap();
        save(&p, "id-a", &["m.gguf".into()], 2, 3, &s, serde_json::json!({})).unwrap();
        assert!(p.with_extension("json.prev").exists());
        assert_eq!(load(&p, "id-a", 2, 3), Some(s.clone()));
        assert_eq!(load(&p, "id-b", 2, 3), None);
        assert_eq!(load(&p, "id-a", 3, 3), None);
        std::fs::write(&p, b"{broken").unwrap();
        assert_eq!(load(&p, "id-a", 2, 3), None);
        // a version-1 file (usage only) is read
        std::fs::write(&p, serde_json::to_vec(&serde_json::json!({"version": 1, "model": "id-a", "layers": 2, "experts": 3, "tokens": 5.0, "score": [0, 1, 2, 3, 4, 5]})).unwrap()).unwrap();
        let v1 = load(&p, "id-a", 2, 3).unwrap();
        assert_eq!(v1.learned.tokens, 5.0);
        assert_eq!(v1.learned.score[5], 5.0);
        assert!(v1.devices.is_empty());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn rotation_drops_old_devices_and_extra_configs() {
        let mut s = Saved::new(1);
        let now = 1_000_000_000;
        s.devices.insert("GPU-old".into(), serde_json::json!({"seen": now - DEVICE_KEEP_SECS - 1}));
        s.devices.insert("GPU-new".into(), serde_json::json!({"seen": now - 10}));
        for i in 0..20u64 {
            s.configs.insert(format!("c{i}"), serde_json::json!({"seen": now - 100 + i}));
        }
        s.rotate(now);
        assert!(!s.devices.contains_key("GPU-old") && s.devices.contains_key("GPU-new"));
        assert_eq!(s.configs.len(), MAX_CONFIGS);
        assert!(s.configs.contains_key("c19") && !s.configs.contains_key("c0"));
    }

    #[test]
    fn names_follow_model_shards_and_overlays() {
        let p = |s: &str| PathBuf::from(s);
        assert_eq!(state_name(&[p("/m/a.gguf")]), "a.gguf");
        assert_eq!(state_name(&[p("/m/a-00001-of-00003.gguf"), p("/m/a-00002-of-00003.gguf"), p("/m/a-00003-of-00003.gguf")]), "a-00001-of-00003.gguf");
        assert_eq!(state_name(&[p("/m/a.gguf"), p("/o/a-uncensored-overlay.gguf")]), "a.gguf+a-uncensored-overlay");
        assert_eq!(state_name(&[p("/m/a-00001-of-00002.gguf"), p("/m/a-00002-of-00002.gguf"), p("/o/ov.gguf")]), "a-00001-of-00002.gguf+ov");
        // next to the model when writable
        let d = std::env::temp_dir().join(format!("dsx-tier-name-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(state_path(&[d.join("x.gguf")]), d.join("x.gguf.tier.json"));
        assert_ne!(state_path(&[d.join("x.gguf"), d.join("ov.gguf")]), state_path(&[d.join("x.gguf")]));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn reset_removes_state_prev_and_buffer() {
        let d = std::env::temp_dir().join(format!("dsx-tier-reset-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let st = d.join("m.gguf.tier.json");
        std::fs::write(&st, b"{}").unwrap();
        std::fs::write(st.with_extension("json.prev"), b"{}").unwrap();
        assert_eq!(reset_files(&st, None, "m.gguf-not-buffered"), 2);
        assert!(!st.exists());
        request_reset();
        assert!(reset_requested());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn only_the_first_model_of_its_shape_claims_tiering() {
        let t = Tier::new(2, 8);
        assert!(!t.claim(3, 8), "other shape");
        assert!(t.claim(2, 8));
        assert!(!t.claim(2, 8), "second model");
    }

    #[test]
    fn picks_hits_and_store_counts() {
        let t = Tier::new(2, 8);
        t.note_picks(1, &[7, 3, 3]);
        let s = t.snapshot_picks();
        assert_eq!(s[8 + 7], 1);
        assert_eq!(s[8 + 3], 2);
        t.note_picks(1, &[99]); // out of range: ignored
        t.note_hit(Hit::File, 2);
        assert_eq!(t.hits()[3], 2);
        t.set_loc(1, 3, LOC_GPU + 1);
        t.note_gpu(1, 3);
        t.note_gpu(1, 4); // not on a GPU per loc: tier count only
        assert_eq!(t.hits()[0], 2);
        assert_eq!(t.store_picks[1].load(Ordering::Relaxed), 1);
        t.note_decode();
        assert!(t.idle_for() <= 1);
    }

    #[test]
    fn plan_moves_hot_off_gpu_against_clearly_colder_same_key_residents() {
        // 1 layer x 8 experts; store 0 swaps by tiering, store 1 is adaptive (left alone)
        let stores = vec![
            StoreInfo { uuid: "GPU-a".into(), name: "a".into(), link_gbs: 3.0, tier_swaps: true },
            StoreInfo { uuid: "GPU-b".into(), name: "b".into(), link_gbs: 24.0, tier_swaps: false },
        ];
        let loc = [LOC_GPU, LOC_GPU, LOC_GPU + 1, LOC_FILE, LOC_ARENA, LOC_FILE, LOC_FILE, LOC_GPU];
        let key = [1, 1, 1, 1, 1, 2, 1, 1];
        let score = [5.0, 100.0, 1.0, 50.0, 30.0, 999.0, 3.0, 20.0];
        let m = plan_moves(8, &score, &loc, &key, &stores, &[true, true], &[], 4);
        // 50 replaces the coldest tier store resident (5), 30 the next (20 > 15: no), 999 has no same-key victim,
        // 3 is below the minimum, store 1's resident (1) is not a tiering victim
        assert_eq!(m, vec![Move { new: (0, 3), store: 0, victim: (0, 0) }]);
        // a move in transit blocks both experts
        let m2 = plan_moves(8, &score, &loc, &key, &stores, &[true, true], &m, 4);
        assert!(m2.iter().all(|x| x.new != (0, 3) && x.victim != (0, 0)));
        // a store on the critical path takes nothing
        assert!(plan_moves(8, &score, &loc, &key, &stores, &[false, true], &[], 4).is_empty());
    }

    #[test]
    fn waits_decide_which_gpu_may_take_cpu_work() {
        let mut w = Waits::default();
        assert_eq!(w.open(2), vec![true, true], "unmeasured: open");
        for _ in 0..WAIT_MIN_SAMPLES {
            w.add(300.0, 0.0); // the main GPU waits for the CPU; the aux GPU is done before it
        }
        assert_eq!(w.open(2), vec![true, true]);
        let mut a = w;
        for _ in 0..200 {
            a.add(300.0, 150.0); // the aux GPU finishes last: it takes no more
        }
        assert_eq!(a.open(2), vec![true, false]);
        let mut c = w;
        for _ in 0..200 {
            c.add(0.0, 0.0); // the CPU is never waited for: no move helps
        }
        assert_eq!(c.open(2), vec![false, false]);
        let t = Tier::new(1, 1);
        t.note_waits(0.3, 0.0);
        assert_eq!(t.waits().samples, 1);
        assert!((t.waits().cpu_us - 300.0).abs() < 1e-9);
    }

    #[test]
    fn prefetch_slots_fill_take_release() {
        let mut mem = vec![0u8; 2 * 64];
        let pf = Prefetch::new(mem.as_mut_ptr(), 64, 2);
        let mv = Move { new: (0, 1), store: 0, victim: (0, 2) };
        let i = pf.claim().unwrap();
        assert!(pf.fill(i, mv, [&[1u8; 10], &[2u8; 20], &[3u8; 30]]));
        assert_eq!(pf.in_transit(), vec![mv]);
        let (j, got) = pf.take().unwrap();
        assert_eq!((j, got), (i, mv));
        let p = pf.parts(j).unwrap();
        assert_eq!((p[0].len(), p[1][0], p[2][29]), (10, 2, 3));
        assert!(pf.take().is_none(), "busy until released");
        pf.release(j);
        assert!(pf.in_transit().is_empty());
        let k = pf.claim().unwrap();
        assert!(!pf.fill(k, mv, [&[0u8; 40], &[0u8; 20], &[0u8; 10]]), "larger than a slot");
        drop(mem);
    }

    #[test]
    fn store_costs_need_all_measured() {
        let mut s = Saved::new(1);
        s.devices.insert("a".into(), serde_json::json!({"expert_us": 40.0}));
        s.devices.insert("b".into(), serde_json::json!({"expert_us": 60.0}));
        assert_eq!(store_costs(&s, &["a".into(), "b".into()]), Some(vec![1.0, 1.5]));
        assert_eq!(store_costs(&s, &["a".into(), "c".into()]), None);
    }

    struct Fake(Vec<Vec<u8>>);
    impl ExpertBytes for Fake {
        fn parts(&self, l: usize, e: usize) -> [&[u8]; 3] {
            let b = &self.0[l * 4 + e];
            [&b[..100], &b[100..300], &b[300..]]
        }
    }

    #[test]
    fn ssd_buffer_put_get_persist_and_refuse_when_full() {
        let d = std::env::temp_dir().join(format!("dsx-ssd-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let lens = vec![[100u32, 200, 4000]; 8];
        let fake = Fake((0..8).map(|i| vec![i as u8 + 1; 4300]).collect());
        let b = SsdBuffer::open(&d, "id", "m.gguf", lens.clone(), 3 * 8192).unwrap(); // entries are 4 KiB aligned: 3 fit
        assert!(b.get(5).is_none());
        assert!(b.put(5, fake.parts(1, 1)).unwrap());
        let g = b.get(5).unwrap();
        assert_eq!((g[0].len(), g[1].len(), g[2].len()), (100, 200, 4000));
        assert!(g[2].iter().all(|&x| x == 6));
        assert!(b.put(2, fake.parts(0, 2)).unwrap());
        assert!(b.put(3, fake.parts(0, 3)).unwrap());
        assert!(!b.put(4, fake.parts(1, 0)).unwrap(), "full");
        b.save_index().unwrap();
        drop(b);
        let b2 = SsdBuffer::open(&d, "id", "m.gguf", lens.clone(), 3 * 8192).unwrap();
        assert!(b2.holds(5) && b2.holds(2) && !b2.holds(4));
        assert!(b2.get(2).unwrap()[0].iter().all(|&x| x == 3));
        let b3 = SsdBuffer::open(&d, "other-model", "m.gguf", lens, 3 * 8192).unwrap();
        assert!(!b3.holds(5), "another model starts empty");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn idle_step_warms_hottest_file_side_first_and_measure_records_devices() {
        let tier: &'static Tier = Box::leak(Box::new(Tier::new(2, 4)));
        // file side: (0,1), (1,2), (0,3); the rest on a GPU or in the arena
        for i in 0..8 {
            tier.loc[i].store(LOC_ARENA, Ordering::Relaxed);
        }
        for (l, e) in [(0, 1), (1, 2), (0, 3)] {
            tier.set_loc(l, e, LOC_FILE);
        }
        tier.set_loc(0, 0, LOC_GPU);
        tier.set_stores(vec![StoreInfo { uuid: "GPU-x".into(), name: "X".into(), link_gbs: 24.6, tier_swaps: true }]);
        let fake = Fake((0..8).map(|i| vec![i as u8; 4300]).collect());
        let mut saved = Saved::new(8);
        saved.learned.score[1 * 4 + 2] = 9.0;
        saved.learned.score[0 * 4 + 1] = 5.0;
        let mut bg = Background {
            tier, state: std::env::temp_dir().join("x.json"), id: "i".into(), files: vec![], top_k: 6, saved,
            model_disk: Some("disk:8:0".into()), model_read_mb_s: Some(180.0), slow_media: true,
        };
        assert_eq!(bg.order()[0], (1, 2));
        assert_eq!(bg.order().len(), 3);
        let mut st = Stats::default();
        let mut warmed = std::collections::HashSet::new();
        if ram_headroom() > 0 {
            let moved = bg.idle_step(&fake, &mut st, &mut warmed);
            assert_eq!(moved, 3 * 4300);
            assert!(warmed.contains(&(1, 2)));
        }
        for _ in 0..10 {
            tier.note_gpu_ms(0, 0.5);
        }
        tier.passes.store(200, Ordering::Relaxed);
        tier.all_passes.store(200, Ordering::Relaxed);
        tier.busy_us.store(10_000_000, Ordering::Relaxed);
        tier.store_picks[0].store(200 * 2 * 3, Ordering::Relaxed); // 3 picks per layer
        bg.measure();
        let d = &bg.saved.devices["GPU-x"];
        assert_eq!(d["name"], "X");
        assert!((d["expert_us"].as_f64().unwrap() - 166.7).abs() < 0.2, "{d}"); // 0.5 ms / 3 picks
        assert_eq!(bg.saved.devices["disk:8:0"]["read_mb_s"], 180.0);
        assert_eq!(bg.saved.configs["GPU-x"]["decode_tok_s"], 20.0);
        let j = bg.stats_json(&st);
        assert!(j["share_from_file"].is_number());
        assert_eq!(j["experts_on_gpus"], 1);
    }
}
