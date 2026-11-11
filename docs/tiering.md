# Automatic expert tiering

dsx places routed experts over the GPUs, a RAM arena and the model file at startup, in the order of `--rank`
([placement](placement.md)). Adaptive GPU stores then swap experts by recent use; every other store keeps the set it
got at startup, and what the server learned about this machine's traffic is gone after a restart.

`nvidia/src/tier.rs` learns which experts this machine uses, measures which device each layer waits for, moves work
toward the devices with room while the server runs, and keeps all of it next to the model for the next start. It needs
no option and adapts to whatever GPUs, RAM and disks a start finds.

## What it does

| When | What |
|---|---|
| every decode pass | the CPU worker counts each layer's picks (GPU-resident ones included) and which tier served them: a GPU store, the arena, the SSD buffer or the file |
| every 64th decode token | one layer (rotating through all of them) is timed with CUDA events: each GPU's expert kernels, and how long the main GPU waits for the CPU and then for each aux GPU |
| every 10 s | a background thread folds the counts into a long-term score per expert (half-life 200k decode tokens) |
| between tokens, at most one per token | a planned move or exchange starts (see below) |
| in every pause of 2 s | the engine thread runs planned moves and exchanges back to back; the background thread warms the page cache with the hottest experts the CPU reads from the file side and, on a slow medium, copies them into an SSD buffer. All of it stops when a request arrives |
| every 2 min, busy or not | the state is saved atomically next to the model; the previous version stays as `.prev` |
| next startup | the learned score is blended into the ranking before placement |

Routing is never changed: only where an expert's bytes are and which device computes it. As with any placement, a
different device can change the floating-point summation order of an expert's output.

## Who is waited for

In every layer the main GPU, the CPU worker and the aux GPUs compute their experts in parallel; the layer is done when
the last of them is. After its own work the main GPU waits for the CPU's experts, then for each aux GPU in turn. Each
wait is how much later that device finished than everything waited for before it. Smoothed over the samples (from 16
on) this says which device is on the critical path, and tiering moves work away from it:

| The main GPU waits | Meaning | What tiering does |
|---|---|---|
| for an aux GPU (> 20 µs after the CPU) | that aux GPU finishes last | **exchanges**: its hottest experts swap places with colder ones on the main GPU |
| for the CPU (> 20 µs), not for aux GPUs | the CPU finishes last | **moves**: hot experts the CPU computes go onto GPUs with a fixed set that are not late |
| for nobody | the main GPU is last or all are even | nothing |

The waits, the CPU's picks per token and decode tokens/s are kept per GPU combination in the state file.

## Moves and exchanges

The background thread plans them and copies the experts into one of four pinned prefetch slots (from the arena, the
SSD buffer or the file). Between two tokens, or in a pause, the engine checks the plan against the current state
(experts where planned, nothing in flight for them, same quant types, room to copy) and starts it with the protocol of
adaptive replacement (`GpuExperts::begin_swap` / `commit_ready`):

- **Move from the CPU** (into a store that keeps a fixed set: the main GPU's with `DSX_ADAPT=0`, aux GPUs on slower
  links): the victim leaves the store's table first and the CPU computes it from the next token on; the new expert's
  bytes go to the victim's slot on the side stream after the current token's kernels and enter the table only when the
  copy has completed.
- **Exchange between GPUs** (a late aux GPU and the main GPU, both able to replace experts): the hot expert leaves the
  aux GPU's table and the main GPU's colder victim leaves the main GPU's table, both before either copy starts. Until
  the copies land, the CPU computes both. Then the hot expert is resident on the main GPU and the victim in the hot
  expert's old slot on the aux GPU. Both stores stay full; no expert is ever resident on two GPUs.

A candidate needs a learned score of at least 8 and a victim of at most half its score. Stores that already swap by
recent use (`DSX_ADAPT`) keep doing so; they take part in exchanges only as the main GPU.

## State

`<model file>.tier.json` next to the model when its directory is writable, else in `$DSX_STATE_DIR`,
`$XDG_STATE_HOME/dsx`, `~/.local/state/dsx` or `/var/tmp/dsx`. With an overlay the name gets `+<overlay stem>`
(`model.gguf+uncensored-overlay.tier.json`): each model and overlay learns on its own. A change of any file (size or
modification time) or of the model's shape starts fresh; a file of the first format (usage only) is read.

```json
{
  "version": 2,
  "model": {"id": "…", "files": ["…/model.gguf", "…/overlay.gguf"]},
  "usage": {"tokens": …, "score": [ … per layer × expert … ]},
  "devices": {
    "GPU-3f2a…": {"name": "…", "link_gbs": …, "expert_layer_ms": …, "picks_per_token": …, "seen": …},
    "disk:8:16": {"read_mb_s": …, "seen": …}
  },
  "configs": {
    "GPU-3f2a…+GPU-9c11…": {"decode_tok_s": …, "cpu_picks_per_token": …, "wait_for_cpu_us": …, "wait_for_aux_us": [ … ], "share_from_file": …, "seen": …}
  },
  "stats": { … }
}
```

- `usage` does not depend on the hardware: the same state serves a start on one GPU or on several, with more or less
  free memory; placement takes whatever stores this start has, in learned order.
- `devices` are keyed by GPU UUID (the same in every `CUDA_VISIBLE_DEVICES` order) and by disk; a GPU missing in one
  start keeps its record.
- Rotation: devices unseen for 90 days are dropped, at most 16 GPU combinations are kept, two state files at most.
- `--tier-reset` removes this model's state, its previous version and its SSD buffer at startup, then runs as usual.

## Budgets and safety

- Page cache: at most 40 GB, never below 16 GB of `MemAvailable`, never closer than 4 GB to the process's memory cgroup
  limit (`memory.max`). Warming and SSD copies run only in pauses: they would compete with the CPU's expert reads for
  memory bandwidth.
- SSD buffer: when the model's medium is rotational or reads below 1000 MB/s (measured with O_DIRECT at startup); in
  the first writable directory on another non-rotational device with more than 20 GB free (`$DSX_CACHE_DIR`, the
  state directory, `$XDG_CACHE_HOME/dsx`, `~/.cache/dsx`, `/var/tmp/dsx`); half its free space, at most 512 GB and at
  most the size of the file-side experts. Entries are appended, synced and only then published.
- GPU memory: one staging buffer (the largest expert) per fixed-set store; nothing else is allocated at run time.
- A failed timing query drops the sample; a failed pause slice stops that round. Only the first model of a process
  claims the tiering state. Every other failure (no writable directory, no SSD, no pinned memory, unreadable state)
  disables only that part.

## What the log shows

Lines of this form (numbers depend on the machine):

```
tier: placement from … learned tokens (…% learned, …% shipped ranking)
experts resident: GPU0 … (… GB), GPU1 … (… GB, static, tier moves, … GB/s link)
tier: … experts on the file side (… GB), model medium … MB/s; state …/model.gguf.tier.json
tier: {"picks":{…},"share_from_file":…,"experts_on_gpus":…,"moves_planned":…,"wait_for_cpu_us":…,"wait_for_aux_us":…,…}
```

## Not done here

- The arena is not reshuffled while the server runs (it is a pinned, CUDA-registered allocation); the page cache and
  the SSD buffer serve the file side instead, and learned use reaches the arena at the next start.
- GPU memory budgets are taken at startup; a GPU added or freed while the server runs is used at the next start.
- Exchanges go between aux GPUs and the main GPU, not between two aux GPUs; the main GPU is still chosen by host link.
- No measurement on a GPU machine is part of this change yet; it needs an A-B-A-B comparison with otherwise
  identical settings.
