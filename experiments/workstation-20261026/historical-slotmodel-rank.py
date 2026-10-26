#!/usr/bin/env python3
"""
Baut aus DSX_LOG_ROUTES-Logs eine DSX-Rangliste ([[layer, expert, count], ...],
absteigend sortiert) und sagt den GPU-/CPU-Pick-Anteil auf getrennten
Eval-Logs voraus, ohne GPU-Lauf.

Nur Standardbibliothek. Liest nur die angegebenen Logs, schreibt nur --out/--stats.

Logformat (bestätigt für handoff.rs): pro Zeile die Layernummer, danach die
Experten-IDs aller Tokens des Durchlaufs, nur Dezimalzahlen, je ein Leerzeichen.
Jede abweichende Zeile bricht ab (strikt, keine Toleranz).

Die CPU-Anteil-Vorhersage ist ein SLOTMODELL (Platzierung nach Expertenanzahl).
Sie bildet experts::assign() (Bytegrößen, Restkapazitäten, Gruppenpriorität,
Füllstand) NICHT nach und ist nur eine grobe Orientierung. Die Zeile
"oracle_slotmodel" ist die beste Häufigkeitssortierung im Slotmodell, keine
Obergrenze für Bytes oder Tokens/s.

Beispiele:
  # 1) Format prüfen
  python3 dsx_rank_from_routes.py --calib calib.log --show 5 --dry-run
  # 2) Rangliste aus Kalibrier-Prompts, nur Decode-Zeilen
  python3 dsx_rank_from_routes.py --calib calib.log --decode-only \
      --out rank-coding.json --stats rank-stats.json
  # 3) zusätzlich Vorhersage auf getrennten Eval-Prompts
  python3 dsx_rank_from_routes.py --calib calib.log --eval eval.log --decode-only \
      --gpu-slots 7318 --out rank-coding.json --stats rank-stats.json
"""
import argparse
import json
import re
import sys

LINE = re.compile(r"\d+(?: \d+)+")  # handoff.rs: Layer, dann Experten-IDs, je ein Leerzeichen


def parse_logs(paths, n_layers, n_exp, k, decode_only, show, label):
    counts = [[0] * n_exp for _ in range(n_layers)]
    st = {"lines": 0, "used_lines": 0, "rejected": 0, "decode_lines": 0,
          "multi_token_lines": 0, "dup_tokens": 0, "picks": 0}
    bad = []
    shown = 0
    for path in paths:
        with open(path, "r", errors="replace") as f:
            for ln, raw in enumerate(f, 1):
                s = raw.rstrip("\n")
                st["lines"] += 1
                if not LINE.fullmatch(s):
                    ok = False
                    nums = []
                else:
                    nums = [int(x) for x in s.split(" ")]
                    ok = len(nums) >= 1 + k
                if ok:
                    layer, picks = nums[0], nums[1:]
                    ok = (0 <= layer < n_layers and len(picks) % k == 0
                          and all(0 <= p < n_exp for p in picks))
                if not ok:
                    st["rejected"] += 1
                    if len(bad) < 5:
                        bad.append(f"{path}:{ln}: {s[:160]}")
                    continue
                ntok = len(picks) // k
                for t in range(ntok):
                    if len(set(picks[t * k:(t + 1) * k])) != k:
                        st["dup_tokens"] += 1
                if ntok == 1:
                    st["decode_lines"] += 1
                else:
                    st["multi_token_lines"] += 1
                if shown < show:
                    print(f"[{label}] {path}:{ln}: layer={layer} tokens={ntok} "
                          f"picks[:12]={picks[:12]}")
                    shown += 1
                if decode_only and ntok != 1:
                    continue
                st["used_lines"] += 1
                for p in picks:
                    counts[layer][p] += 1
                st["picks"] += len(picks)
    if st["lines"] == 0:
        sys.exit(f"FEHLER [{label}]: keine Zeilen gelesen.")
    if st["rejected"]:
        print(f"FEHLER [{label}]: {st['rejected']} von {st['lines']} Zeilen ungültig "
              f"(erwartet: nur Ganzzahlen mit je einem Leerzeichen, Layer < {n_layers}, "
              f"Experten < {n_exp}, Vielfaches von {k} Picks). Beispiele:",
              file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        sys.exit(2)
    if st["dup_tokens"]:
        print(f"WARNUNG [{label}]: {st['dup_tokens']} Token-Gruppen ohne {k} verschiedene "
              f"Experten. Picks evtl. nicht tokenweise gruppiert; Zählung bleibt gültig, "
              f"--decode-only-Erkennung aber prüfen.", file=sys.stderr)
    if st["picks"] == 0:
        sys.exit(f"FEHLER [{label}]: keine Picks gezählt (zu streng gefiltert?).")
    return counts, st


def default_order(n_layers, n_exp):
    # experts::place(None) laut Quelltextbefund: Expertennummer außen, Layer innen.
    return [(l, e) for e in range(n_exp) for l in range(n_layers)]


def rank_order(counts):
    n_layers, n_exp = len(counts), len(counts[0])
    return sorted(((l, e) for l in range(n_layers) for e in range(n_exp)),
                  key=lambda le: (-counts[le[0]][le[1]], le[0], le[1]))


def share_in_top(order, slots, ev):
    gpu = set(order[:slots])
    n_layers = len(ev)
    tot = [sum(ev[l]) for l in range(n_layers)]
    on_gpu = [sum(c for e, c in enumerate(ev[l]) if (l, e) in gpu) for l in range(n_layers)]
    T = sum(tot)
    return {
        "gpu_share": sum(on_gpu) / T,
        "cpu_share": 1 - sum(on_gpu) / T,
        "cpu_share_layers_0_19": 1 - sum(on_gpu[:20]) / max(1, sum(tot[:20])),
        "cpu_share_layers_20_39": 1 - sum(on_gpu[20:]) / max(1, sum(tot[20:])),
        "cpu_share_per_layer": [round(1 - on_gpu[l] / tot[l], 4) if tot[l] else None
                                for l in range(n_layers)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", nargs="+", required=True, help="Route-Logs der Kalibrier-Prompts")
    ap.add_argument("--eval", nargs="*", default=[], help="Route-Logs getrennter Eval-Prompts")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--k", type=int, default=6, help="Picks pro Token")
    ap.add_argument("--decode-only", action="store_true",
                    help="nur Zeilen mit genau einem Token zählen (Prefill ist im Log nicht markiert)")
    ap.add_argument("--gpu-slots", type=int, nargs="+", default=[1022, 7318],
                    help="GPU-Kapazität in Experten (Default: RTX allein, RTX+CMP)")
    ap.add_argument("--show", type=int, default=0, help="erste N geparste Zeilen anzeigen")
    ap.add_argument("--dry-run", action="store_true", help="nur parsen und prüfen")
    ap.add_argument("--out", help="Ranglisten-JSON für DSX --rank")
    ap.add_argument("--stats", help="Statistik-JSON")
    a = ap.parse_args()

    cal, cst = parse_logs(a.calib, a.layers, a.experts, a.k, a.decode_only, a.show, "calib")
    print(f"[calib] Zeilen {cst['lines']}, genutzt {cst['used_lines']}, "
          f"Decode-Zeilen {cst['decode_lines']}, Mehrtoken-Zeilen {cst['multi_token_lines']}, "
          f"Picks {cst['picks']}")
    if a.dry_run:
        return

    order = rank_order(cal)
    total = cst["picks"]
    cum, cov = 0, {}
    marks = sorted(set(a.gpu_slots) | {len(order) // 10, len(order) // 4, len(order) // 2})
    for i, (l, e) in enumerate(order, 1):
        cum += cal[l][e]
        if i in marks:
            cov[i] = cum / total
    print("[calib] Anteil der Picks in den heißesten N Experten (Kalibrierdaten selbst):")
    for n in marks:
        print(f"  N={n:5d}: {cov[n]:.3f}")
    zero = sum(1 for l, e in order if cal[l][e] == 0)
    print(f"[calib] Experten ohne einzigen Pick: {zero} von {len(order)} "
          f"(ihre Reihenfolge ist willkürlich; mehr Kalibriertext nötig, falls viele)")

    stats = {"calib": cst, "calib_coverage": cov, "unpicked_experts": zero,
             "calib_files": a.calib, "eval_files": a.eval, "decode_only": a.decode_only,
             "per_layer_calib_picks": [sum(r) for r in cal]}

    if a.eval:
        ev, est = parse_logs(a.eval, a.layers, a.experts, a.k, a.decode_only, 0, "eval")
        stats["eval"] = est
        stats["prediction"] = {}
        orders = {"default_place_none": default_order(a.layers, a.experts),
                  "calib_rank": order,
                  "oracle_slotmodel": rank_order(ev)}
        print("[eval] CPU-Pick-Anteil im SLOTMODELL (Expertenanzahl statt Bytes; bildet "
              "experts::assign() nicht nach, nur Orientierung):")
        for slots in a.gpu_slots:
            stats["prediction"][slots] = {}
            for name, o in orders.items():
                r = share_in_top(o, slots, ev)
                stats["prediction"][slots][name] = r
                print(f"  slots={slots:5d} {name:24s} CPU {r['cpu_share']:.3f}  "
                      f"(L0-19 {r['cpu_share_layers_0_19']:.3f}, "
                      f"L20-39 {r['cpu_share_layers_20_39']:.3f})")
        print("  Hinweis: Abweichungen vom gemessenen CPU-Anteil sind im Slotmodell und bei "
              "anderen Prompts zu erwarten; sie belegen keinen Parserfehler.")

    if a.out:
        with open(a.out, "w") as f:
            json.dump([[l, e, cal[l][e]] for l, e in order], f)
        print(f"Rangliste geschrieben: {a.out} ({len(order)} Einträge, absteigend)")
    if a.stats:
        with open(a.stats, "w") as f:
            json.dump(stats, f, indent=1)
        print(f"Statistik geschrieben: {a.stats}")


if __name__ == "__main__":
    main()
