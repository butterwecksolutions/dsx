# Automatic expert tiering

dsx places routed experts over the GPUs, a RAM arena and the model file at startup, in the order of `--rank`
([placement](placement.md)). Adaptive GPU stores then swap experts by recent use; every other store keeps the set it
got at startup, and what the server learned about this machine's traffic used to be gone after a restart.

`nvidia/src/tier.rs` learns which experts this machine uses, moves them toward the fastest place while the server
runs, measures the devices it runs on and keeps all of it next to the model for the next start. It needs no option.

## What it does

| When | What |
|---|---|
| every decode pass | the CPU worker counts each layer's picks (GPU-resident ones included) and which tier served them: a GPU store, the arena, the SSD buffer or the file |
| every 10 s | a background thread folds the counts into a long-term score per expert (half-life 200k decode tokens) |
| while decoding, at most one per token | a hot expert on no GPU replaces a clearly colder resident (at most half its score, same quant types) of a GPU store that otherwise keeps a fixed set |
| every 64th token | one layer's expert kernels are timed on every GPU (CUDA events): cost per pick per GPU |
| idle (no decode for 30 s) | the page cache is warmed with the hottest experts the CPU reads from the file side, within a RAM budget |
| idle, model on a slow medium | those experts are copied into an SSD buffer on another, faster device; the CPU reads them from there at once |
| idle, every 5 min | the state is saved atomically; the previous version stays as `.prev` |
| next startup | the learned score is blended into the ranking before placement, and with measured GPU costs the hottest experts go preferably to the faster GPU |

Routing is never changed: only where an expert's bytes are and which device computes it. As with any placement, a
different device can change the floating-point summation order of an expert's output.

### Run-time moves

Stores that already swap by recent use (`DSX_ADAPT`, the main GPU's by default, fast-linked aux GPUs) keep doing so
and are left alone. Every other store with the fast expert layout — the main GPU's with `DSX_ADAPT=0`, aux GPUs on
slower links — gets one staging buffer and takes tiering's moves:

1. The background thread picks the move and copies the new expert into a pinned prefetch slot (from the arena, the
   SSD buffer or the file). At most four experts are in transit.
2. Between two tokens the engine checks the move against the current state (new expert on no GPU and not in flight,
   victim still resident, same types) and starts it with the same protocol as adaptive replacement
   (`GpuExperts::begin_swap`): the victim leaves the store's table first and the CPU computes it from the next token
   on; the new expert's bytes go to the victim's slot on the side stream after the current token's kernels.
3. Only when that copy has completed does the new expert enter the table (`commit_swaps`); its prefetch slot is freed.

No expert is ever computed from a partially written slot, and no arena copy of the victim is needed: the CPU reads it
from the arena, the SSD buffer or the file, which is why fixed sets did not swap before (`DSX_ADAPT=0` exists for
machines whose RAM cannot hold those copies).

## State

`<model file>.tier.json` next to the model when its directory is writable, else in `$DSX_STATE_DIR`,
`$XDG_STATE_HOME/dsx`, `~/.local/state/dsx` or `/var/tmp/dsx`. With an overlay the name gets `+<overlay stem>`
(`model.gguf+uncensored-overlay.tier.json`): each model and overlay learns on its own. A change of any file (size or
modification time) or of the model's shape starts fresh.

```json
{
  "version": 2,
  "model": {"id": "…", "files": ["…/model.gguf", "…/overlay.gguf"]},
  "usage": {"tokens": …, "score": [ … per layer × expert … ]},
  "devices": {
    "GPU-3f2a…": {"name": "…", "link_gbs": …, "expert_us": …, "samples": …, "seen": …},
    "disk:8:16": {"read_mb_s": …, "seen": …}
  },
  "configs": {
    "GPU-3f2a…+GPU-9c11…": {"decode_tok_s": …, "share_from_file": …, "seen": …}
  },
  "stats": { … }
}
```

- `usage` does not depend on the hardware: the same state serves a start on one GPU or on several, with more or less
  free memory; placement takes whatever stores this start has, in learned order.
- `devices` are keyed by GPU UUID (the same in every `CUDA_VISIBLE_DEVICES` order) and by disk; a GPU that is missing
  in one start keeps its record.
- Rotation: devices unseen for 90 days are dropped, at most 16 GPU combinations are kept, two state files at most.
- `--tier-reset` removes this model's state, its previous version and its SSD buffer at startup, then runs as usual.

## Budgets and safety

- Page cache: at most 40 GB, never below 16 GB of `MemAvailable`, never closer than 4 GB to the process's memory cgroup
  limit (`memory.max`). Warming stops as soon as a request arrives.
- SSD buffer: when the model's medium is rotational or reads below 1000 MB/s (measured with O_DIRECT at startup); in
  the first writable directory on another non-rotational device with more than 20 GB free (`$DSX_CACHE_DIR`, the
  state directory, `$XDG_CACHE_HOME/dsx`, `~/.cache/dsx`, `/var/tmp/dsx`); half its free space, at most 512 GB and at
  most the size of the file-side experts. Entries are appended, synced and only then published.
- GPU memory: one staging buffer (the largest expert) per store taking moves; nothing else is allocated at run time.
- Every failure (no writable directory, no SSD, no pinned memory, unreadable state) only disables that part.

## What the log shows

Lines of this form (numbers depend on the machine):

```
tier: measured expert cost per GPU (fastest = 1): 1.00, …
experts resident: GPU0 … (… GB), GPU1 … (… GB, static, tier moves, … GB/s link)
tier: placement from … learned tokens (…% learned, …% shipped ranking)
tier: … experts on the file side (… GB), model medium … MB/s; state …/model.gguf.tier.json
tier: {"picks":{"gpu":…,"arena":…,"ssd":…,"file":…},"share_from_file":…,"experts_on_gpus":…,"moves_planned":…}
```

`share_from_file` is the fraction of the CPU's expert reads that still go to the file side; the goal is to drive it
toward zero. The same figures are in the state file.

## Not done here

- The arena is not reshuffled while the server runs (it is a pinned, CUDA-registered allocation); the page cache and
  the SSD buffer serve the file side instead, and learned use reaches the arena at the next start.
- GPU memory budgets are taken at startup; a GPU added or freed while the server runs is used at the next start.
- The main GPU is still chosen by host link (it carries the dense weights, prefill and the adaptive cache).
- No measurement on a GPU machine is part of this change yet; it needs an A-B-A-B comparison with otherwise
  identical settings.
