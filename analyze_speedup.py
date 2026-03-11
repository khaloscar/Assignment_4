import os
import math
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt

OUTDIR = "plots"


def mean_std_sem(xs):
    """
    Returns:
        mean : arithmetic mean
        std  : sample standard deviation (ddof=1)
        sem  : standard error of the mean = std / sqrt(n)
    """
    n = len(xs)
    if n == 0:
        raise ValueError("Cannot compute statistics of empty list")

    mean = sum(xs) / n

    if n > 1:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        std = math.sqrt(var)
        sem = std / math.sqrt(n)
    else:
        std = 0.0
        sem = 0.0

    return mean, std, sem


def read_timings(filename="timings_speedup.txt"):
    """
    Expected line format:
        runtime  N  version  nsteps  nthreads
    """
    groups = defaultdict(list)
    problems = set()

    with open(filename, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{filename}:{lineno}: expected 5 fields, got {len(parts)}: {line}"
                )

            runtime = float(parts[0])
            N = int(parts[1])
            version = parts[2]
            nsteps = int(parts[3])
            nthreads = int(parts[4])

            groups[(version, N, nsteps, nthreads)].append(runtime)
            problems.add((N, nsteps))

    if not groups:
        raise ValueError(f"No timing data found in {filename}")

    return groups, sorted(problems, key=lambda p: (p[0], p[1]))


def build_stats(groups):
    stats = {}
    for key, times in groups.items():
        mean, std, sem = mean_std_sem(times)
        stats[key] = {
            "times": list(times),
            "n": len(times),
            "mean": mean,
            "std": std,
            "sem": sem,
            "min": min(times),
            "max": max(times),
        }
    return stats


def build_timing_table(stats):
    lines = []
    lines.append("=" * 118)
    lines.append(
        f"{'Version':<14} {'N':>8} {'Steps':>8} {'Threads':>8} "
        f"{'Mean (s)':>12} {'Std (s)':>12} {'SEM (s)':>12} "
        f"{'Min (s)':>12} {'Max (s)':>12} {'n':>6}"
    )
    lines.append("=" * 118)

    keys_sorted = sorted(stats.keys(), key=lambda k: (k[1], k[2], k[0], k[3]))
    for version, N, nsteps, nthreads in keys_sorted:
        s = stats[(version, N, nsteps, nthreads)]
        lines.append(
            f"{version:<14} {N:>8} {nsteps:>8} {nthreads:>8} "
            f"{s['mean']:>12.6f} {s['std']:>12.6f} {s['sem']:>12.6f} "
            f"{s['min']:>12.6f} {s['max']:>12.6f} {s['n']:>6}"
        )

    lines.append("")
    lines.append("Note: Std is the sample standard deviation over repeated runs.")
    lines.append("      SEM is the standard error of the mean = Std / sqrt(n).")
    lines.append("")
    return "\n".join(lines)


def compute_speedups(stats, baseline_mode="seq"):
    """
    baseline_mode:
        - 'self' : each version compared to its own 1-thread mean
        - '<version_name>' : all versions compared to that version's 1-thread mean
                             for the same problem size (N, nsteps)
    """
    all_versions = sorted({k[0] for k in stats.keys()})
    all_problems = sorted({(k[1], k[2]) for k in stats.keys()}, key=lambda p: (p[0], p[1]))

    speedups = {}
    warnings = []
    versions_used = set()
    problems_used = set()
    problem_versions = defaultdict(set)
    problem_threads = defaultdict(set)

    for N, nsteps in all_problems:
        versions_here = sorted(
            {v for (v, N0, s0, nt) in stats.keys() if N0 == N and s0 == nsteps}
        )
        threads_here = sorted(
            {nt for (v, N0, s0, nt) in stats.keys() if N0 == N and s0 == nsteps}
        )

        if baseline_mode == "self":
            for version in versions_here:
                base_key = (version, N, nsteps, 1)
                if base_key not in stats:
                    warnings.append(
                        f"Warning: missing self-baseline {(version, N, nsteps, 1)}; "
                        f"skipping speedups for version '{version}' at N={N}, nsteps={nsteps}"
                    )
                    continue

                Tb = stats[base_key]["mean"]
                sem_b = stats[base_key]["sem"]

                for nthreads in threads_here:
                    key = (version, N, nsteps, nthreads)
                    if key not in stats:
                        continue

                    Tp = stats[key]["mean"]
                    sem_p = stats[key]["sem"]

                    S = Tb / Tp

                    if nthreads == 1:
                        sem_S = 0.0
                    else:
                        rel_b = sem_b / Tb if Tb > 0 else 0.0
                        rel_p = sem_p / Tp if Tp > 0 else 0.0
                        sem_S = S * math.sqrt(rel_b * rel_b + rel_p * rel_p)

                    eff = 100.0 * S / nthreads

                    speedups[key] = {
                        "speedup": S,
                        "speedup_sem": sem_S,
                        "efficiency_pct": eff,
                        "baseline_version": version,
                    }
                    versions_used.add(version)
                    problems_used.add((N, nsteps))
                    problem_versions[(N, nsteps)].add(version)
                    problem_threads[(N, nsteps)].add(nthreads)

        else:
            baseline_version = baseline_mode
            base_key = (baseline_version, N, nsteps, 1)

            if base_key not in stats:
                warnings.append(
                    f"Warning: missing baseline {base_key}; skipping all speedups "
                    f"for N={N}, nsteps={nsteps}"
                )
                continue

            Tb = stats[base_key]["mean"]
            sem_b = stats[base_key]["sem"]

            for version in versions_here:
                for nthreads in threads_here:
                    key = (version, N, nsteps, nthreads)
                    if key not in stats:
                        continue

                    Tp = stats[key]["mean"]
                    sem_p = stats[key]["sem"]

                    S = Tb / Tp

                    if version == baseline_version and nthreads == 1:
                        sem_S = 0.0
                    else:
                        rel_b = sem_b / Tb if Tb > 0 else 0.0
                        rel_p = sem_p / Tp if Tp > 0 else 0.0
                        sem_S = S * math.sqrt(rel_b * rel_b + rel_p * rel_p)

                    eff = 100.0 * S / nthreads

                    speedups[key] = {
                        "speedup": S,
                        "speedup_sem": sem_S,
                        "efficiency_pct": eff,
                        "baseline_version": baseline_version,
                    }
                    versions_used.add(version)
                    problems_used.add((N, nsteps))
                    problem_versions[(N, nsteps)].add(version)
                    problem_threads[(N, nsteps)].add(nthreads)

    versions_used = sorted(versions_used)
    problems_used = sorted(problems_used, key=lambda p: (p[0], p[1]))

    lines = []

    if baseline_mode == "self":
        lines.append("Baseline mode: each version compared to its own 1-thread mean")
    else:
        lines.append(
            f"Baseline mode: all versions compared to '{baseline_mode}' @ 1 thread "
            "for the same problem size"
        )

    if warnings:
        lines.append("")
        lines.extend(warnings)
        lines.append("")

    for N, nsteps in problems_used:
        lines.append(f"Problem: N={N}, nsteps={nsteps}")
        lines.append("=" * 104)
        lines.append(
            f"{'Version':<14} {'Threads':>8} {'Mean (s)':>12} {'SEM (s)':>12} "
            f"{'Speedup':>12} {'SEM(S)':>12} {'Efficiency':>12} {'Baseline':>12}"
        )
        lines.append("=" * 104)

        versions_here = sorted(problem_versions[(N, nsteps)])
        threads_here = sorted(problem_threads[(N, nsteps)])

        for version in versions_here:
            for nthreads in threads_here:
                key = (version, N, nsteps, nthreads)
                if key not in speedups:
                    continue

                s = stats[key]
                sp = speedups[key]

                lines.append(
                    f"{version:<14} {nthreads:>8} "
                    f"{s['mean']:>12.6f} {s['sem']:>12.6f} "
                    f"{sp['speedup']:>12.4f} {sp['speedup_sem']:>12.4f} "
                    f"{sp['efficiency_pct']:>11.1f}% {sp['baseline_version']:>12}"
                )
            lines.append("-" * 104)
        lines.append("")

    speedup_table = "\n".join(lines)

    best_lines = []
    best_lines.append("=" * 84)
    best_lines.append(
        f"{'N':>8} {'Steps':>8} {'Threads':>8} {'Best Version':<14} "
        f"{'Mean (s)':>12} {'Speedup':>12} {'SEM(S)':>12}"
    )
    best_lines.append("=" * 84)

    for N, nsteps in problems_used:
        threads_here = sorted(problem_threads[(N, nsteps)])
        versions_here = sorted(problem_versions[(N, nsteps)])

        for nthreads in threads_here:
            candidates = []
            for version in versions_here:
                key = (version, N, nsteps, nthreads)
                if key in speedups:
                    candidates.append((version, stats[key]["mean"]))

            if not candidates:
                continue

            best_version, best_mean = min(candidates, key=lambda x: x[1])
            best_key = (best_version, N, nsteps, nthreads)
            sp = speedups[best_key]

            best_lines.append(
                f"{N:>8} {nsteps:>8} {nthreads:>8} {best_version:<14} "
                f"{best_mean:>12.6f} {sp['speedup']:>12.4f} {sp['speedup_sem']:>12.4f}"
            )

    best_lines.append("")
    best_table = "\n".join(best_lines)

    return speedups, versions_used, problems_used, warnings, speedup_table, best_table


def should_plot_version(version, baseline_mode):
    return baseline_mode == "self" or version != baseline_mode


def get_problem_styles(problems):
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "p", "x", "*"]

    styles = {}
    for i, problem in enumerate(problems):
        styles[problem] = {
            "linestyle": linestyles[i % len(linestyles)],
            "marker": markers[i % len(markers)],
        }
    return styles


def plot_speedups_same(speedups, versions, problems, baseline_mode="seq"):
    if not speedups:
        print("No speedup data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    thread_counts = sorted({k[3] for k in speedups.keys()})
    max_threads = max(thread_counts)

    ax.plot(
        [1, max_threads],
        [1, max_threads],
        "k--",
        alpha=0.35,
        linewidth=1.5,
        label="Ideal linear speedup",
    )

    plot_versions = [v for v in versions if should_plot_version(v, baseline_mode)]
    problem_styles = get_problem_styles(problems)

    for N, nsteps in problems:
        style = problem_styles[(N, nsteps)]

        for version in plot_versions:
            xs = []
            ys = []

            for nthreads in thread_counts:
                key = (version, N, nsteps, nthreads)
                if key not in speedups:
                    continue
                xs.append(nthreads)
                ys.append(speedups[key]["speedup"])

            if xs:
                label = f"{version} (N={N}, steps={nsteps})"
                ax.plot(
                    xs,
                    ys,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=2.2,
                    markersize=6,
                    label=label,
                )

    ax.set_xlabel("Number of threads", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_title(f"Speedup vs Threads (baseline: {baseline_mode})", fontsize=14)
    ax.set_xticks(thread_counts)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max_threads + 0.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    plt.tight_layout()
    outfile = os.path.join(OUTDIR, "speedup_plot_same.png")
    plt.savefig(outfile, dpi=150)
    print(f"Plot saved to {outfile}")
    plt.show()


def plot_speedups_sep(speedups, versions, problems, baseline_mode="seq"):
    if not speedups:
        print("No speedup data to plot.")
        return

    nplots = len(problems)
    ncols = 2 if nplots > 1 else 1
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    plot_versions = [v for v in versions if should_plot_version(v, baseline_mode)]
    markers = ["o", "s", "^", "D", "v", "p", "x", "*"]

    for ax, (N, nsteps) in zip(axes, problems):
        problem_keys = [
            k for k in speedups.keys()
            if k[1] == N and k[2] == nsteps
        ]
        if not problem_keys:
            ax.set_visible(False)
            continue

        threads_here = sorted({k[3] for k in problem_keys})
        max_threads = max(threads_here)

        ax.plot(
            [1, max_threads],
            [1, max_threads],
            "k--",
            alpha=0.35,
            linewidth=1.5,
            label="Ideal linear speedup",
        )

        for idx, version in enumerate(plot_versions):
            xs = []
            ys = []

            for nthreads in threads_here:
                key = (version, N, nsteps, nthreads)
                if key not in speedups:
                    continue
                xs.append(nthreads)
                ys.append(speedups[key]["speedup"])

            if xs:
                ax.plot(
                    xs,
                    ys,
                    linestyle="-",
                    marker=markers[idx % len(markers)],
                    linewidth=2.2,
                    markersize=6,
                    label=version,
                )

        ax.set_title(f"N={N}, nsteps={nsteps}")
        ax.set_xlabel("Number of threads")
        ax.set_ylabel("Speedup")
        ax.set_xticks(threads_here)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, max_threads + 0.5)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)

    for ax in axes[len(problems):]:
        ax.set_visible(False)

    plt.suptitle(f"Speedup vs Threads (baseline: {baseline_mode})", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    outfile = os.path.join(OUTDIR, "speedup_plot_sep.png")
    plt.savefig(outfile, dpi=150)
    print(f"Plot saved to {outfile}")
    plt.show()


def plot_speedups(speedups, versions, problems, baseline_mode="seq", plot_mode="same"):
    if plot_mode == "sep":
        plot_speedups_sep(speedups, versions, problems, baseline_mode=baseline_mode)
    else:
        plot_speedups_same(speedups, versions, problems, baseline_mode=baseline_mode)


def save_and_print(text, filename):
    print(text)
    filepath = os.path.join(OUTDIR, filename)
    with open(filepath, "w") as f:
        f.write(text + "\n")
    print(f"  -> Saved to {filepath}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze timing results")
    parser.add_argument(
        "--baseline",
        default="seq",
        help=(
            "Baseline mode. Default: 'seq'. "
            "Use '--baseline self' to compare each version to its own 1-thread run, "
            "or '--baseline v1' / '--baseline v2' / etc. to compare all versions "
            "against that version's 1-thread runtime for the same problem size."
        ),
    )
    parser.add_argument(
        "--plot",
        choices=["same", "sep"],
        default="same",
        help=(
            "Plot layout mode. "
            "'same' = all problem sizes on one plot (default), "
            "'sep' = separate subplot per problem size."
        ),
    )
    parser.add_argument(
        "--timings",
        default="timings_speedup.txt",
        help="Path to timings file (default: timings_speedup.txt)",
    )
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    groups, problems = read_timings(args.timings)
    stats = build_stats(groups)

    print("\n--- Timing Results ---\n")
    timing_table = build_timing_table(stats)
    save_and_print(timing_table, "timing_table.txt")

    print(f"--- Speedup Analysis (baseline: {args.baseline}) ---\n")
    speedups, versions, problems_used, warnings, speedup_table, best_table = compute_speedups(
        stats, baseline_mode=args.baseline
    )
    save_and_print(speedup_table, "speedup_table.txt")

    print("--- Best Version per Thread Count ---\n")
    save_and_print(best_table, "best_version_table.txt")

    plot_speedups(
        speedups,
        versions,
        problems_used,
        baseline_mode=args.baseline,
        plot_mode=args.plot,
    )
