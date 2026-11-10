//! Automatic expert tiering across media, learned from use and kept across restarts.
//!
//! Tiers, fastest first: GPU stores (placed at startup by the ranking; adaptive stores swap at run time), the RAM
//! arena (filled at startup in ranking order), the page cache (file pages kept in RAM), an SSD buffer (only when the
//! model file sits on a rotational disk) and the model file itself. Nothing here changes which experts the router
//! picks; it changes where their bytes come from.
//!
//! * **Usage.** The CPU worker counts every decode pick (`note_picks`). A background thread folds the counts into a
//!   long-term score with a slow half-life (`HALF_LIFE_TOKENS`) and saves it next to its statistics in a state file
//!   (`state_path`), atomically, every few minutes and only while idle.
//! * **Startup.** `blend_rank` mixes the shipped ranking (`--rank`) with the learned score, so the next start puts the
//!   experts this machine actually uses on the GPUs and into the arena. The learned share grows with the number of
//!   learned tokens (at most `ALPHA_MAX`).
//! * **Idle.** With no decode for `IDLE_SECS`, the thread warms the page cache with the hottest experts left in the file
//!   (within a RAM budget that respects the process's memory cgroup) and, if the model is on a rotational disk, copies
//!   them into an SSD buffer that the CPU worker reads from at once (`ssd_get`) and that survives restarts.
//! * **Measurement.** Picks per tier, read speeds of the model file and the SSD buffer, page-cache warm bytes.
//!
//! No option: the state lives in `$DSX_STATE_DIR`, else `$XDG_STATE_HOME/dsx`, else `~/.local/state/dsx`, else
//! `/var/tmp/dsx`. Any failure only disables the affected part; inference never depends on it.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub const HALF_LIFE_TOKENS: f64 = 200_000.0;
pub const ALPHA_FULL_TOKENS: f64 = 50_000.0;
pub const ALPHA_MAX: f64 = 0.8;
pub const IDLE_SECS: u64 = 30;
const SAVE_EVERY_SECS: u64 = 300;
const WARM_MAX_BYTES: u64 = 40 << 30;
const MEM_RESERVE_BYTES: u64 = 16 << 30;
const CGROUP_RESERVE_BYTES: u64 = 4 << 30;
const SSD_MIN_FREE_BYTES: u64 = 20 << 30;
const SSD_MAX_BYTES: u64 = 512 << 30;
const STEP_BYTES: u64 = 512 << 20;
const STATE_VERSION: u32 = 1;

/// Which tier served an expert's bytes for the CPU (GPU-resident picks are counted separately).
#[derive(Clone, Copy)]
pub enum Hit {
    Gpu = 0,
    Arena = 1,
    Ssd = 2,
    File = 3,
}

pub struct Tier {
    pub nl: usize,
    pub ne: usize,
    /// monotonic decode-pick counts per [layer * E + expert] (the CPU worker is the only writer)
    picks: Vec<AtomicU32>,
    hits: [AtomicU64; 4],
    last_decode: AtomicU64,
    ssd: OnceLock<SsdBuffer>,
    stop: AtomicBool,
}

static GLOBAL: OnceLock<&'static Tier> = OnceLock::new();

/// The process-wide tier state (created by `init`), or None when tiering is off for this run.
pub fn global() -> Option<&'static Tier> {
    GLOBAL.get().copied()
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
            last_decode: AtomicU64::new(0),
            ssd: OnceLock::new(),
            stop: AtomicBool::new(false),
        }
    }

    /// Register as the process-wide instance (first call wins).
    pub fn init(nl: usize, ne: usize) -> &'static Tier {
        GLOBAL.get_or_init(|| Box::leak(Box::new(Tier::new(nl, ne))))
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

    /// Called once per decode pass: marks the server busy (no warming or copying meanwhile).
    #[inline]
    pub fn note_decode(&self) {
        self.last_decode.store(now_secs(), Ordering::Relaxed);
    }

    pub fn idle_for(&self) -> u64 {
        now_secs().saturating_sub(self.last_decode.load(Ordering::Relaxed))
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

/// Identity of the model file the state belongs to: canonical path, size, modification time.
pub fn model_id(path: &Path) -> String {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    match std::fs::metadata(path) {
        Ok(m) => format!("{}|{}|{}", canon.display(), m.len(), m.mtime()),
        Err(_) => canon.display().to_string(),
    }
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
    let probe = d.join(format!(".probe-{}", std::process::id()));
    let ok = File::create(&probe).is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

/// `<state dir>/<model file name>.tier.json`
pub fn state_path(model: &Path) -> PathBuf {
    let name: String = model
        .file_name()
        .map(|n| n.to_string_lossy().chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '_' }).collect())
        .unwrap_or_else(|| "model".into());
    state_dir().join(format!("{name}.tier.json"))
}

/// Read the learned state if it belongs to this model and shape; None otherwise (a new model starts fresh).
pub fn load(path: &Path, id: &str, nl: usize, ne: usize) -> Option<Learned> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    if v["version"].as_u64()? != STATE_VERSION as u64 || v["model"].as_str()? != id || v["layers"].as_u64()? as usize != nl || v["experts"].as_u64()? as usize != ne {
        return None;
    }
    let score: Vec<f64> = v["score"].as_array()?.iter().map(|x| x.as_f64().unwrap_or(0.0).max(0.0)).collect();
    (score.len() == nl * ne).then(|| Learned { score, tokens: v["tokens"].as_f64().unwrap_or(0.0).max(0.0) })
}

/// Write the state atomically (temp file + rename); the previous version stays as `.prev` (rotation: fixed size,
/// two files at most).
pub fn save(path: &Path, id: &str, nl: usize, ne: usize, l: &Learned, stats: serde_json::Value) -> std::io::Result<()> {
    let v = serde_json::json!({
        "version": STATE_VERSION, "model": id, "layers": nl, "experts": ne, "tokens": l.tokens,
        "score": l.score.iter().map(|s| (s * 1000.0).round() / 1000.0).collect::<Vec<_>>(), "stats": stats,
        "saved": now_secs(),
    });
    let tmp = path.with_extension("json.tmp");
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

/// Copies of hot experts on an SSD when the model file sits on a rotational disk. One preallocated (sparse) file,
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

/// Where to put the SSD buffer: the first writable directory on a non-rotational device with room.
pub fn ssd_dir(model: &Path) -> Option<PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let cands = [
        env("DSX_CACHE_DIR"),
        Some(state_dir()),
        env("XDG_CACHE_HOME").map(|p| p.join("dsx")),
        env("HOME").map(|p| p.join(".cache/dsx")),
        Some(PathBuf::from("/var/tmp/dsx")),
    ];
    let model_dev = std::fs::metadata(model).ok().map(|m| m.dev());
    cands.into_iter().flatten().find(|d| {
        std::fs::create_dir_all(d).is_ok()
            && writable(d)
            && rotational(d) == Some(false)
            && std::fs::metadata(d).ok().map(|m| m.dev()) != model_dev
            && free_bytes(d).is_some_and(|f| f > SSD_MIN_FREE_BYTES)
    })
}

// ------------------------------------------------------------------------------------------------ background work

/// What the background thread needs from the model: each expert's three byte slices (in the mapped model file).
pub trait ExpertBytes: Sync {
    fn parts(&self, l: usize, e: usize) -> [&[u8]; 3];
}

pub struct Background {
    pub tier: &'static Tier,
    pub state: PathBuf,
    pub id: String,
    pub top_k: usize,
    /// experts the CPU computes from the file (not on a GPU, not in the arena), hottest first is decided at run time
    pub file_only: Vec<(usize, usize)>,
    pub learned: Learned,
    pub model_hdd: bool,
}

#[derive(Default, Debug)]
pub struct Stats {
    pub warm_bytes: u64,
    pub warm_secs: f64,
    pub ssd_bytes: u64,
    pub ssd_secs: f64,
    pub saves: u64,
}

impl Background {
    fn order(&self) -> Vec<(usize, usize)> {
        let ne = self.tier.ne;
        let mut v = self.file_only.clone();
        v.sort_by(|a, b| self.learned.score[b.0 * ne + b.1].total_cmp(&self.learned.score[a.0 * ne + a.1]));
        v
    }

    fn stats_json(&self, st: &Stats) -> serde_json::Value {
        let h = self.tier.hits();
        let tot = (h.iter().sum::<u64>()).max(1) as f64;
        let rate = |b: u64, s: f64| if s > 0.05 && b >= 64 << 20 { Some((b as f64 / s / 1e6).round()) } else { None };
        serde_json::json!({
            "picks": {"gpu": h[0], "arena": h[1], "ssd": h[2], "file": h[3]},
            "share_from_file": ((h[3] as f64 / tot) * 1000.0).round() / 1000.0,
            "file_only_experts": self.file_only.len(),
            "page_cache_warm_gb": (st.warm_bytes as f64 / 1e9 * 10.0).round() / 10.0,
            "model_read_mb_s": rate(st.warm_bytes, st.warm_secs),
            "ssd_buffer": self.tier.ssd.get().map(|s| serde_json::json!({"path": s.path(), "used_gb": (s.used() as f64 / 1e9 * 10.0).round() / 10.0,
                                                                         "write_mb_s": rate(st.ssd_bytes, st.ssd_secs)})),
            "model_on_hdd": self.model_hdd,
            "learned_tokens": self.learned.tokens.round(),
            "learned_share": (self.learned.alpha() * 1000.0).round() / 1000.0,
        })
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
                            advise(b, MADV_DONTNEED); // the HDD pages are no longer needed in RAM
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

    /// The loop: fold usage every 10 s, work and save while idle, until the tier is stopped.
    pub fn run(mut self, m: &'static dyn ExpertBytes) {
        let mut last = self.tier.snapshot_picks();
        let mut st = Stats::default();
        let mut warmed = std::collections::HashSet::new();
        let mut last_save = std::time::Instant::now();
        let mut dirty = false;
        let mut last_report = std::time::Instant::now();
        while !self.tier.stop.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_secs(10));
            let now = self.tier.snapshot_picks();
            let delta: Vec<u32> = now.iter().zip(&last).map(|(a, b)| a.wrapping_sub(*b)).collect();
            last = now;
            if delta.iter().any(|&d| d > 0) {
                self.learned.fold(&delta, self.top_k, self.tier.nl);
                dirty = true;
            }
            if self.tier.idle_for() < IDLE_SECS {
                continue;
            }
            self.idle_step(m, &mut st, &mut warmed);
            if dirty && last_save.elapsed().as_secs() >= SAVE_EVERY_SECS {
                if save(&self.state, &self.id, self.tier.nl, self.tier.ne, &self.learned, self.stats_json(&st)).is_ok() {
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
        let _ = save(&self.state, &self.id, self.tier.nl, self.tier.ne, &self.learned, self.stats_json(&st));
    }
}

/// Set up the SSD buffer for a model on a rotational disk (no-op otherwise); returns whether one is active.
pub fn attach_ssd(tier: &Tier, model: &Path, lens: Vec<[u32; 3]>, file_only_bytes: u64) -> bool {
    if rotational(model) != Some(true) {
        return false;
    }
    let Some(dir) = ssd_dir(model) else {
        eprintln!("tier: model on a rotational disk, but no SSD directory with room found (DSX_CACHE_DIR)");
        return false;
    };
    let cap = (free_bytes(&dir).unwrap_or(0) / 2).min(SSD_MAX_BYTES).min(file_only_bytes.next_multiple_of(1 << 30)).next_multiple_of(4096);
    if cap < 1 << 30 {
        return false;
    }
    let name = model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    match SsdBuffer::open(&dir, &model_id(model), &name, lens, cap) {
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
        // experts missing from the shipped ranking still appear exactly once
        let r = blend_rank(&base[..1], &none, 2, 2);
        let mut s = r.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 4);
        assert_eq!(r[0], (0, 0));
    }

    #[test]
    fn state_roundtrip_rotation_and_identity() {
        let d = std::env::temp_dir().join(format!("dsx-tier-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("m.tier.json");
        let mut l = Learned::new(6);
        l.score[3] = 2.5;
        l.tokens = 7.0;
        save(&p, "id-a", 2, 3, &l, serde_json::json!({})).unwrap();
        save(&p, "id-a", 2, 3, &l, serde_json::json!({})).unwrap();
        assert!(p.with_extension("json.prev").exists());
        assert_eq!(load(&p, "id-a", 2, 3), Some(l.clone()));
        assert_eq!(load(&p, "id-b", 2, 3), None);
        assert_eq!(load(&p, "id-a", 3, 3), None);
        std::fs::write(&p, b"{broken").unwrap();
        assert_eq!(load(&p, "id-a", 2, 3), None);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn picks_and_hits_are_counted() {
        let t = Tier::new(2, 8);
        t.note_picks(1, &[7, 3, 3]);
        let s = t.snapshot_picks();
        assert_eq!(s[8 + 7], 1);
        assert_eq!(s[8 + 3], 2);
        t.note_picks(1, &[99]); // out of range: ignored
        t.note_hit(Hit::File, 2);
        assert_eq!(t.hits()[3], 2);
        t.note_decode();
        assert!(t.idle_for() <= 1);
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
    fn idle_step_warms_hottest_first_within_one_step() {
        let tier: &'static Tier = Box::leak(Box::new(Tier::new(2, 4)));
        let fake = Fake((0..8).map(|i| vec![i as u8; 4300]).collect());
        let mut learned = Learned::new(8);
        learned.score[1 * 4 + 2] = 9.0;
        learned.score[0 * 4 + 1] = 5.0;
        let bg = Background { tier, state: std::env::temp_dir().join("x.json"), id: "i".into(), top_k: 6, file_only: vec![(0, 1), (1, 2), (0, 3)], learned, model_hdd: false };
        assert_eq!(bg.order()[0], (1, 2));
        let mut st = Stats::default();
        let mut warmed = std::collections::HashSet::new();
        if ram_headroom() > 0 {
            let moved = bg.idle_step(&fake, &mut st, &mut warmed);
            assert_eq!(moved, 3 * 4300);
            assert!(warmed.contains(&(1, 2)));
        }
        let j = bg.stats_json(&st);
        assert!(j["share_from_file"].is_number());
    }
}
