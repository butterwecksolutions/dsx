# Automatic expert tiering

dsx places routed experts over the GPUs, a RAM arena and the model file at startup, in the order of `--rank`
([placement](placement.md)). Adaptive GPU stores then swap experts by recent use. Everything after that is fixed for the
run, and what the server learned about this machine's traffic is gone after a restart.

`nvidia/src/tier.rs` keeps that knowledge and acts on it. It needs no flag.

## What it does

| When | What | Where the bytes come from afterwards |
|---|---|---|
| every decode pass | the CPU worker counts each layer's picks (`note_picks`), GPU-resident ones included, and which tier served the CPU's experts | unchanged |
| every 10 s | a background thread folds the counts into a long-term score (half-life 200k decode tokens) | unchanged |
| idle (no decode for 30 s) | warms the page cache with the hottest experts that live only in the file, within a RAM budget | page cache instead of the disk |
| idle, model file on a rotational disk | copies the hottest file-only experts into an SSD buffer; the CPU worker reads them from there at once | SSD instead of the HDD |
| idle, every 5 min | saves the score and statistics (`<model>.tier.json`, written atomically, previous version kept as `.prev`) | — |
| next startup | blends the learned score into the ranking before placement | GPUs and arena hold what this machine uses |

The learned share of the startup ranking grows with the learned tokens, from 0 to at most 80 % at 50k tokens; the
shipped calibration ranking keeps the rest. With no learned tokens the order is exactly `--rank`. Routing is never
changed. As with any ranking, placement can change floating-point summation order between devices.

## Budgets and safety

- Page cache: at most 40 GB, never below 16 GB of `MemAvailable`, and never closer than 4 GB to the process's memory
  cgroup limit (`memory.max`). Warming stops as soon as a request arrives.
- SSD buffer: only when the model file is on a rotational disk; half of the free space of the first writable
  non-rotational directory (`$DSX_CACHE_DIR`, the state directory, `$XDG_CACHE_HOME/dsx`, `~/.cache/dsx`, `/var/tmp/dsx`),
  at most 512 GB and at most the size of the file-only experts. Entries are appended, synced and only then published;
  the index is rewritten atomically. A buffer for another model file is discarded.
- State: `$DSX_STATE_DIR`, else `$XDG_STATE_HOME/dsx`, else `~/.local/state/dsx`, else `/var/tmp/dsx`. A state file for
  another model (path, size or modification time differ) or another shape is ignored.
- Every failure (no writable directory, no SSD, unreadable state) only disables that part; inference does not depend
  on it.

## What the log shows

```
tier: placement from 61234 learned tokens (80% learned, 20% shipped ranking)
tier: 4421 experts in the file (70.3 GB) on a rotational disk; usage state /home/u/.local/state/dsx/model.gguf.tier.json
tier: SSD buffer /home/u/.local/state/dsx/model.gguf.ssd-buffer (180 GB, 42.0 GB held)
tier: {"picks":{"gpu":…,"arena":…,"ssd":…,"file":…},"share_from_file":0.012,"page_cache_warm_gb":38.5,"model_read_mb_s":…}
```

`share_from_file` is the fraction of the CPU's expert reads that still go to the model file; the goal of tiering is to
drive it toward zero. The same numbers are in the state file under `stats`.

## Not done here

- The arena is not reshuffled while the server runs (it is a pinned, CUDA-registered allocation); learned use reaches
  it at the next start.
- GPU budgets are not changed at run time. The startup link measurement still decides each GPU's role.
- No measurement on a GPU machine is part of this change yet; it needs an A-B-A-B comparison of a learned against
  the shipped ranking with otherwise identical settings.
