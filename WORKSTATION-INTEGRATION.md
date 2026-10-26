# Workstation integration — 2026-10-26

This branch combines the contributions for workstation testing. It is not a production release or a modification of the installer r42. Pin a reviewed commit and cache source/dependencies for offline use.

| Development | Integrated state | Upstream review | Evidence |
|---|---|---|---|
| Native Q2_K / IQ3_XXS format support, IQ3 SIMD, Q2_K stage 1 | Enabled source | #12 | Core fixture test and 19 CPU library tests; CPU benchmark; historical workstation TPS |
| Route hotness calibration | quant/rank_from_routes.py | #13 | Four Python tests; exact reproduction of all 15,360 historical rank entries from 61,120 route lines |
| Request timing | Opt-in DSX_TIMING_LOG, refined counters | #14 (draft) | Rust syntax and source/API inspection; actual CUDA build/request tests pending |
| Q2_K stage 2 / s16 minimum | Archived patch only, NOT applied | Not proposed as default | Applies to stage-1 source; all 19 CPU tests pass, including tolerance-based comparison; float results are not bit-identical; no validated additional full-model gain |
| Historical request timing | Archived patch only | Superseded by #14 | Kept to reproduce historical development; do not apply on top of integrated timing |
| Historical slot-model rank utility | Archived only | Replaced by focused #13 CLI | Slot predictions are not byte-placement or TPS predictions; original script is not the production CLI |

The optional Q2_K stage 2 test replaces the bit-identity assertion with a tolerance comparison. Do not describe it as lossless or bit-identical. It requires separate quality/performance A/B validation before consideration for activation.

The exact development archives remain local; published docs contain sanitized request rates, source archive hashes and methodology. Workstation-specific GPU UUIDs, protection services, deployment scripts, credentials and model files are intentionally not bundled into upstream PRs. Expert placement and adaptive caching were already upstream functionality; our contribution is the calibration workflow and measurements, not invention of those mechanisms.

## Remaining acceptance

CUDA compilation, synthetic GPU regressions and current-tree full-model A/B on the i9-10900X + RTX 3090 + CMP 170HX must pass before replacing the deployed binary. Preserve the existing binary and GPU watchdog. Current measured historical results are documented in docs/native-quant-workstation-history.md and docs/route-hotness-calibration.md; they are not a guarantee for this combined revision.

The runtime-source integration does not itself change setup r42 or establish a complete airgap recovery cache. That requires installer source pinning, dependency/toolchain caching and separate offline acceptance.
