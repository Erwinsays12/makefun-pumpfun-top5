#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: a Monte Carlo forecaster for the top-5 chart.

Reads $MAKEFUN_INPUT_DIR/history.ndjson + task.json and writes
$MAKEFUN_OUTPUT_DIR/forecast.json.

Approach
--------
Everything is measured from the history the house hands over, so the same
program runs on the 2-hour practice miniature and on the final's 11-hour
history with its reads 7-13 hours after the cut:

* Market dynamics: per-venue (bonding curve vs graduated pool) drift and
  volatility of log10 market cap, estimated from consecutive-mark changes,
  shrunk toward conservative priors; a per-hour "dump" hazard for sudden
  3x-20x collapses, also estimated from the marks. Coins that vanish from
  a full chart enter as censored steps down to just under the chart's
  floor, so the estimates are not fooled by survivorship.
* Incumbents: every plausible coin (seen near the cut, or once a real
  contender) is evolved as a random walk in log-cap: its own recent slope
  (shrunk by how much data backs it) fades with a ~1.5 h time constant into
  an age- and venue-dependent prior drift; curve coins that cross the
  measured graduation cap switch to pool dynamics.
* Newcomers: launches arrive at the rate the history's `eligible` counter
  implies. A small, measured share of launches are "contenders" that reach
  at least half of the #5 cap; they keep arriving after the cut as a Poisson
  process (widened by a factor for the coins born before the cut that never
  charted and moon late), rocket to a peak drawn from the history's
  contender peaks within tens of minutes, then decay with a measured
  half-life. They are never named: their probability mass stays in each
  place's unnamed share, which is exactly how the scorer pays for a place
  a coin the history never ranked takes.
* The field is ranked at every answer read; place frequencies and place
  cap histograms over thousands of seeded simulations become the forecast.
  A fixed seed and sorted iteration make two runs byte-identical.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 987123
MICRO = 1_000_000


def parse_ts(text):
    """ISO 8601 UTC -> epoch seconds (float)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def median(xs, default):
    if not xs:
        return default
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean_sd(xs, default_mean, default_sd):
    if len(xs) < 2:
        return default_mean, default_sd
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(max(v, 1e-9))


def shrink(observed, prior, n, strength):
    """Weighted blend of an observed statistic toward a prior."""
    if n <= 0:
        return prior
    w = n / (n + strength)
    return w * observed + (1.0 - w) * prior


def poisson(rng, lam):
    if lam <= 0:
        return 0
    if lam > 40:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    # Knuth
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while p > limit:
        k += 1
        p *= rng.random()
    return k - 1


def load_inputs(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    marks.sort(key=lambda m: m["at"])
    return task, marks


def build_series(marks):
    """mint -> list of (epoch, log10 cap in SOL, venue, complete, created_epoch)."""
    series = {}
    for m in marks:
        t = parse_ts(m["at"])
        for r in m["rows"]:
            logcap = math.log10(int(r["market_cap_lamports"])) - 9.0
            series.setdefault(r["mint"], []).append(
                (t, logcap, r["venue"], bool(r["complete"]), parse_ts(r["created_at"]))
            )
    for obs in series.values():
        obs.sort()
    return series


def step_stats(series, marks, cadence_h):
    """Per-venue drift/hour, vol/sqrt-hour and dump hazard/hour from mark steps.

    Steps between two chart appearances are only half the story: a coin that
    vanishes from a full 200-row chart fell below that chart's floor, so each
    disappearance is added as a censored step down to just under the floor.
    Without them every statistic is biased toward the survivors.
    """
    raw = {"curve": {"d": [], "z": [], "dump": 0, "steps": 0, "zsteps": 0},
           "pool": {"d": [], "z": [], "dump": 0, "steps": 0, "zsteps": 0}}

    def add(key, dt, dc, exact=True):
        s = raw[key]
        s["d"].append(dc / dt)
        if exact:  # censored drops bound the move, they do not measure vol
            s["z"].append(dc / math.sqrt(dt))
            s["zsteps"] += 1
        s["steps"] += 1
        if dc <= -0.5:
            s["dump"] += 1

    mark_floor = []
    mark_mints = []
    for m in marks:
        if m["rows"]:
            mark_floor.append(min(math.log10(int(r["market_cap_lamports"])) - 9.0 for r in m["rows"]))
            mark_mints.append({r["mint"] for r in m["rows"]})
        else:
            mark_floor.append(None)
            mark_mints.append(set())
    mark_time = [parse_ts(m["at"]) for m in marks]
    for mint, obs in series.items():
        by_mark = {}
        for (t, c, v, done, created) in obs:
            # locate the mark this observation came from (times are unique)
            for mi, mt in enumerate(mark_time):
                if abs(mt - t) < 1.0:
                    by_mark[mi] = (t, c, v)
                    break
        idxs = sorted(by_mark)
        for a, b in zip(idxs, idxs[1:]):
            t1, c1, v1 = by_mark[a]
            t2, c2, _ = by_mark[b]
            dt = (t2 - t1) / 3600.0
            if 0 < dt <= 2.5 * cadence_h:
                add("pool" if v1 == "pool" else "curve", dt, c2 - c1)
            elif b == a + 1:
                pass  # gap too long; nothing to infer
        # censored disappearance: present at mark a, absent at full mark a+1
        for a in idxs:
            b = a + 1
            if b < len(marks) and mark_floor[b] is not None and len(marks[b]["rows"]) >= 200 \
                    and mint not in mark_mints[b]:
                t1, c1, v1 = by_mark[a]
                dt = (mark_time[b] - t1) / 3600.0
                if 0 < dt <= 2.5 * cadence_h:
                    add("pool" if v1 == "pool" else "curve", dt,
                        max(mark_floor[b] - 0.10 - c1, -1.0), exact=False)
    stats = {}
    priors = {
        "curve": {"drift": -0.08, "sigma": 0.30, "hazard": 0.06},
        "pool": {"drift": 0.00, "sigma": 0.28, "hazard": 0.035},
    }
    for key, s in raw.items():
        pr = priors[key]
        drift_obs, _ = mean_sd(s["d"], pr["drift"], 1.0)
        _, sigma_obs = mean_sd(s["z"], 0.0, pr["sigma"])
        stats[key] = {
            "drift": clamp(shrink(drift_obs, pr["drift"], s["steps"], 120), -0.50, 0.25),
            "sigma": clamp(shrink(sigma_obs, pr["sigma"], s["zsteps"], 60), 0.12, 0.60),
            "hazard": clamp(
                shrink(s["dump"] / max(s["steps"], 1) / cadence_h if s["steps"] else pr["hazard"],
                       pr["hazard"], s["steps"], 120),
                0.005, 0.35),
            "n": s["steps"],
        }
    return stats


def graduation_threshold(series):
    """The cap at which curves complete: curve caps never exceed it, so the
    largest curve cap observed is the honest estimate. (Pool rows below it
    are graduates that already fell back, not evidence about the threshold.)"""
    curve_max = None
    pool_first = []
    for obs in series.values():
        seen_pool = False
        for (_, c, v, _, _) in obs:
            if v == "curve":
                curve_max = c if curve_max is None else max(curve_max, c)
            elif v == "pool" and not seen_pool:
                pool_first.append(c)
                seen_pool = True
    if curve_max is not None:
        return clamp(curve_max, 2.2, 3.0)
    if pool_first:
        return clamp(median(pool_first, 2.5), 2.2, 3.0)
    return 2.5


def contender_stats(series, last5_log, eligible_last):
    """Coins that ever reached half of the #5 cap: how many, how high, how fast."""
    thr = last5_log - math.log10(2.0)
    peaks, ttps, half_lives = [], [], []
    for obs in series.values():
        pk = max(obs, key=lambda o: o[1])
        pk_t, pk_c = pk[0], pk[1]
        if pk_c < thr:
            continue
        peaks.append(pk_c)
        created = obs[0][4]
        ttps.append(max((pk_t - created) / 3600.0, 0.02))
        for (t, c, _, _, _) in obs:
            if t > pk_t and pk_c - c > 0.05:
                half_lives.append((t - pk_t) / 3600.0 * math.log10(2.0) / (pk_c - c))
                break
    n = len(peaks)
    p_cont = clamp(n / max(eligible_last, 1), 0.0008, 0.06) if n else 0.004
    mu, sd = mean_sd(peaks, last5_log + 0.1, 0.18)
    # Observed peaks are right-censored (young contenders were still rising
    # at the cut) and thin-tailed in small samples: blend the mean toward
    # the chart's own top peaks and floor the spread, more when n is small.
    all_peaks = sorted((max(o[1] for o in obs) for obs in series.values()), reverse=True)
    if all_peaks:
        top_mean = sum(all_peaks[:5]) / min(5, len(all_peaks))
        w_top = 8.0 / (n + 8.0)
        mu = (1.0 - w_top) * mu + w_top * top_mean
    sd = max(sd, 0.22) if n < 30 else max(sd, 0.15)
    return {
        "p": p_cont,
        "mu": mu,
        "sd": clamp(sd, 0.12, 0.45),
        "ttp": clamp(shrink(median(ttps, 0.35), 0.35, n, 4), 0.05, 2.0),
        "hl": clamp(shrink(median(half_lives, 1.5), 1.5, len(half_lives), 6), 0.5, 5.0),
        "thr": thr,
        "n": n,
    }


def launch_rate(marks):
    if len(marks) < 2:
        return 600.0
    first, last = marks[0], marks[-1]
    dt_h = (parse_ts(last["at"]) - parse_ts(first["at"])) / 3600.0
    if dt_h <= 0:
        return 600.0
    de = last["counts"]["eligible"] - first["counts"]["eligible"]
    return clamp(de / dt_h, 30.0, 20000.0)


def select_incumbents(series, cut, last5_log, last_mark_mints):
    """Plausible place-takers: seen near the cut, in the last mark, or once strong."""
    picked = []
    for mint, obs in series.items():
        last_t, last_c, last_v, last_done, created = obs[-1]
        peak = max(o[1] for o in obs)
        recent = (cut - last_t) <= 45 * 60
        if not (recent or mint in last_mark_mints or peak >= last5_log - 0.45):
            continue
        if peak < last5_log - 2.0:
            continue
        # own slope over the last stretch of observations, log10 per hour
        slope = 0.0
        win = [o for o in obs if last_t - o[0] <= 90 * 60]
        if len(win) >= 2 and win[-1][0] > win[0][0]:
            slope = clamp((win[-1][1] - win[0][1]) / ((win[-1][0] - win[0][0]) / 3600.0), -2.0, 2.0)
        w = min(1.0, len(win) / 5.0)
        stale_h = max(0.0, (cut - last_t) / 3600.0)
        picked.append({
            "mint": mint,
            "start": last_c - 0.12 * min(stale_h, 3.0),
            "peak": peak,
            "slope": slope,
            "w": w,
            "pool": last_v == "pool" or last_done,
            "age_h": max(0.0, (cut - created) / 3600.0),
            "last": last_c,
        })
    picked.sort(key=lambda c: (-c["peak"], -c["last"], c["mint"]))
    return picked[:700]


def simulate(task, marks, series, stats, grad_thr, cont, lam_h, incumbents, last5_log, debug):
    cut = parse_ts(task["history_until"])
    horizons = sorted((parse_ts(r) - cut) / 3600.0 for r in task["reads"])
    # evaluation grid: reads plus substeps so no gap exceeds 0.75 h
    points = [0.0]
    read_flag = [horizons and horizons[0] <= 0.0]
    for h in horizons:
        if h <= points[-1] + 1e-12:
            continue
        t = points[-1]
        while t < h - 1e-12:
            t = min(h, t + 0.75)
            points.append(t)
            read_flag.append(abs(t - h) < 1e-12)
    n_points = len(points)
    n_inc = len(incumbents)
    work = max(1, (n_inc + 60) * n_points)
    n_sims = int(clamp(2.2e7 / work, 1200, 8000))

    places = task["places"]
    edges = task["size"]["edges"]
    n_bins = len(edges) + 1
    place_count = [dict() for _ in range(places)]
    newc_count = [0] * places
    bin_count = [[0] * n_bins for _ in range(places)]
    total_reads = 0

    starts = [c["start"] for c in incumbents]
    is_pool0 = [c["pool"] for c in incumbents]
    # drift(t) = prior + (drift0 - prior) * exp(-t / 1.5): own momentum fades
    drift_prior = []
    drift_start = []
    sigma_of = []
    hazard_of = []
    for c in incumbents:
        key = "pool" if c["pool"] else "curve"
        prior = stats[key]["drift"] - 0.05 * math.log10(1.0 + c["age_h"])
        d0 = c["w"] * c["slope"] + (1.0 - c["w"]) * prior
        drift_prior.append(prior)
        drift_start.append(clamp(d0, -0.7, 0.4))
        sigma_of.append(stats[key]["sigma"])
        hazard_of.append(stats[key]["hazard"])
    sigma_pool, sigma_curve = stats["pool"]["sigma"], stats["curve"]["sigma"]
    hazard_pool, hazard_curve = stats["pool"]["hazard"], stats["curve"]["hazard"]
    # The unknown stream is wider than post-cut births alone: it also stands
    # for coins born before the cut that never charted in the history and
    # moon during the read window (in the practice, 3 of the last read's
    # top 5 were absent from the history, only 2 of them strictly post-cut).
    lam_cont = lam_h * cont["p"] * 1.4

    rng = random.Random(SEED)
    for _ in range(n_sims):
        logs = list(starts)
        pooled = list(is_pool0)
        newcomers = []  # (birth_t, peak_log, ttp_h, hl_h)
        prev_t = points[0]
        for pi in range(n_points):
            t = points[pi]
            dt = t - prev_t
            if dt > 0:
                fade = math.exp(-prev_t / 1.5)
                for i in range(n_inc):
                    drift = drift_prior[i] + (drift_start[i] - drift_prior[i]) * fade
                    # excess cap bleeds: the higher a coin sits above the
                    # chart's #5 level, the harder it is pulled back
                    drift -= 0.22 * max(0.0, logs[i] - last5_log)
                    sig = sigma_pool if pooled[i] else sigma_curve
                    haz = hazard_pool if pooled[i] else hazard_curve
                    x = logs[i] + drift * dt + sig * math.sqrt(dt) * rng.gauss(0.0, 1.0)
                    if rng.random() < haz * dt:
                        x -= rng.uniform(0.4, 1.3)
                    if not pooled[i] and x >= grad_thr:
                        pooled[i] = True
                    logs[i] = x
                n_new = poisson(rng, lam_cont * dt)
                for _j in range(n_new):
                    peak = clamp(rng.gauss(cont["mu"], cont["sd"]), cont["thr"], cont["mu"] + 3 * cont["sd"])
                    ttp = clamp(math.exp(rng.gauss(math.log(cont["ttp"]), 0.55)), 0.03, 3.0)
                    hl = clamp(rng.gauss(cont["hl"], cont["hl"] * 0.35), 0.4, 8.0)
                    newcomers.append((prev_t + dt * rng.random(), peak, ttp, hl))
            if read_flag[pi]:
                total_reads += 1
                field = [(logs[i], i) for i in range(n_inc)]
                for (birth, peak, ttp, hl) in newcomers:
                    age = t - birth
                    if age < 0:
                        continue
                    if age <= ttp:
                        lc = peak + 0.7 * math.log10(max(age / ttp, 1e-3))
                    else:
                        lc = peak - math.log10(2.0) * (age - ttp) / hl
                    field.append((lc + rng.gauss(0.0, 0.09), -1))
                field.sort(key=lambda e: (-e[0], e[1]))
                for k in range(places):
                    if k >= len(field):
                        break
                    lc, idx = field[k]
                    if idx >= 0:
                        mint = incumbents[idx]["mint"]
                        place_count[k][mint] = place_count[k].get(mint, 0) + 1
                    else:
                        newc_count[k] += 1
                    b = 0
                    while b < len(edges) and lc >= edges[b]:
                        b += 1
                    bin_count[k][b] += 1
            prev_t = t
    if debug:
        for k in range(places):
            sys.stderr.write(
                f"place {k + 1}: newcomer share {newc_count[k] / max(total_reads, 1):.3f}, "
                f"named coins {len(place_count[k])}, sims {n_sims}\n")
    return place_count, newc_count, bin_count, total_reads, n_sims


def build_output(task, place_count, bin_count, total_reads, incumbents_by_mint, last5_log):
    places = task["places"]
    edges = task["size"]["edges"]
    n_bins = len(edges) + 1
    total = max(total_reads, 1)
    probabilities = []
    per_place = []
    for k in range(places):
        items = sorted(
            ((mint, c / total) for mint, c in place_count[k].items()),
            key=lambda e: (-e[1], e[0]))
        per_place.append(dict(items))
        rows = 0
        for mint, p in items:
            if p < 0.0015 or rows >= 60:
                break
            # a mild shrink toward the unnamed share: the simulation's
            # parameters are themselves estimates, and log scoring punishes
            # overconfidence in the named coins harder than it rewards it
            q = int(p * 0.92 * MICRO) / MICRO
            if q > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": q})
                rows += 1
        # p=0 rows: stagnant mid-chart coins that cannot hold the place; they
        # leave the unnamed share to the plausible coins and the unborn.
        named = {row["mint"] for row in probabilities if row["place"] == k + 1}
        zeros = [c for c in incumbents_by_mint.values()
                 if c["mint"] not in named
                 and c["peak"] < last5_log - math.log10(2.0)
                 and c["last"] <= last5_log - 0.85
                 and c["age_h"] >= 0.75]
        zeros.sort(key=lambda c: (-c["last"], c["mint"]))
        for c in zeros:
            if rows >= 100:
                break
            probabilities.append({"mint": c["mint"], "place": k + 1, "p": 0})
            rows += 1
    sizes = []
    for k in range(places):
        raw = [bin_count[k][b] / total for b in range(n_bins)]
        mixed = [0.95 * r + 0.05 / n_bins for r in raw]
        mixed = [max(p, 0.0002) for p in mixed]
        s = sum(mixed)
        if s > 0.999:
            mixed = [p * 0.999 / s for p in mixed]
        for b, p in enumerate(mixed):
            q = int(p * MICRO) / MICRO
            if q > 0:
                sizes.append({"place": k + 1, "bin": b, "p": q})
    top5 = []
    used = set()
    for k in range(places):
        best, best_p = None, -1.0
        for mint, p in per_place[k].items():
            if mint not in used and p > best_p:
                best, best_p = mint, p
        if best is None:  # fall back to global popularity among the unnamed
            for mint, p in sorted(
                    ((m, sum(per_place[j].get(m, 0.0) for j in range(places)))
                     for m in incumbents_by_mint),
                    key=lambda e: (-e[1], e[0])):
                if mint not in used:
                    best = mint
                    break
        top5.append(best)
        used.add(best)
    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    debug = bool(os.environ.get("MAKEFUN_DEBUG"))
    task, marks = load_inputs(input_dir)
    if not marks:
        raise SystemExit("history holds no ok mark")
    series = build_series(marks)
    cut = parse_ts(task["history_until"])
    cadence_h = task.get("cadence_minutes", 10) / 60.0
    stats = step_stats(series, marks, cadence_h)
    grad_thr = graduation_threshold(series)
    last_rows = marks[-1]["rows"]
    caps = sorted((int(r["market_cap_lamports"]) for r in last_rows), reverse=True)
    last5_log = math.log10(caps[min(4, len(caps) - 1)]) - 9.0
    cont = contender_stats(series, last5_log, marks[-1]["counts"]["eligible"])
    lam_h = launch_rate(marks)
    last_mark_mints = {r["mint"] for r in last_rows}
    incumbents = select_incumbents(series, cut, last5_log, last_mark_mints)
    if debug:
        sys.stderr.write(f"stats={stats}\ngrad_thr={grad_thr:.3f} last5_log={last5_log:.3f}\n")
        sys.stderr.write(f"cont={cont}\nlaunch_rate/h={lam_h:.1f} incumbents={len(incumbents)}\n")
    place_count, newc_count, bin_count, total_reads, n_sims = simulate(
        task, marks, series, stats, grad_thr, cont, lam_h, incumbents, last5_log, debug)
    by_mint = {c["mint"]: c for c in incumbents}
    out = build_output(task, place_count, bin_count, total_reads, by_mint, last5_log)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(out) + "\n")
    if debug:
        sys.stderr.write(f"sims={n_sims} reads scored={total_reads}\n")


if __name__ == "__main__":
    main()
