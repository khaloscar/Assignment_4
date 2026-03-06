import os
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

OUTDIR = "plots"

def read_timings(filename="timings.txt"):
    groups = defaultdict(list)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            time = float(parts[0])
            N = int(parts[1])
            version = parts[2]
            nsteps = int(parts[3])
            nthreads = int(parts[4])
            key = (version, N, nsteps, nthreads)
            groups[key].append(time)
    return groups

def build_timing_table(groups):
    lines = []
    lines.append("=" * 85)
    lines.append(f"{'Version':<14} {'N':>6} {'Steps':>6} {'Threads':>8} {'Mean (s)':>10} {'Std (s)':>10} {'Min (s)':>10} {'n':>5}")
    lines.append("=" * 85)
    for key in sorted(groups.keys()):
        version, N, nsteps, nthreads = key
        times = groups[key]
        n = len(times)
        mean = sum(times) / n
        std = (sum((t - mean) ** 2 for t in times) / n) ** 0.5
        min_t = min(times)
        lines.append(f"{version:<14} {N:>6} {nsteps:>6} {nthreads:>8} {mean:>10.4f} {std:>10.4f} {min_t:>10.4f} {n:>5}")
    lines.append("")
    return "\n".join(lines)

def compute_speedups(groups, baseline_mode="seq"):
    means = {}
    for key, times in groups.items():
        version, N, nsteps, nthreads = key
        means[(version, nthreads)] = sum(times) / len(times)

    versions = sorted(set(k[0] for k in means.keys()))
    thread_counts = sorted(set(k[1] for k in means.keys()))

    speedups = {}
    header_lines = []

    if baseline_mode == "self":
        header_lines.append("Baseline: each version's own 1-thread run\n")
        parallel_versions = versions
        for version in versions:
            base_key = (version, 1)
            if base_key not in means:
                header_lines.append(f"Warning: no 1-thread baseline for {version}, skipping")
                continue
            base_time = means[base_key]
            for nthreads in thread_counts:
                key = (version, nthreads)
                if key in means:
                    speedups[key] = base_time / means[key]
    else:
        seq_version = baseline_mode
        seq_key = (seq_version, 1)
        if seq_key not in means:
            seq_candidates = {k: v for k, v in means.items() if k[0] == seq_version}
            if seq_candidates:
                seq_key = min(seq_candidates.keys(), key=lambda k: k[1])
                header_lines.append(f"Warning: no 1-thread seq found, using {seq_key}")
            else:
                header_lines.append(f"Warning: no '{seq_version}' version found, using first version with 1 thread")
                for v in versions:
                    if (v, 1) in means:
                        seq_key = (v, 1)
                        break
        base_time = means[seq_key]
        header_lines.append(f"Baseline: {seq_key[0]} @ {seq_key[1]} thread(s) = {base_time:.4f}s\n")

        parallel_versions = [v for v in versions if v != seq_version]
        for version in parallel_versions:
            for nthreads in thread_counts:
                key = (version, nthreads)
                if key in means:
                    speedups[key] = base_time / means[key]

    # Speedup table
    speedup_lines = list(header_lines)
    speedup_lines.append("=" * 65)
    speedup_lines.append(f"{'Version':<14} {'Threads':>8} {'Mean (s)':>10} {'Speedup':>10} {'Efficiency':>10}")
    speedup_lines.append("=" * 65)
    for version in parallel_versions:
        for nthreads in thread_counts:
            key = (version, nthreads)
            if key in speedups:
                s = speedups[key]
                eff = s / nthreads * 100
                speedup_lines.append(f"{version:<14} {nthreads:>8} {means[key]:>10.4f} {s:>10.3f} {eff:>9.1f}%")
        speedup_lines.append("-" * 65)
    speedup_lines.append("")

    # Best version per thread count
    best_lines = []
    best_lines.append("=" * 50)
    best_lines.append(f"{'Threads':>8} {'Best Version':<14} {'Mean (s)':>10} {'Speedup':>10}")
    best_lines.append("=" * 50)
    for nthreads in thread_counts:
        best_version = None
        best_time = float('inf')
        best_speedup = 0
        for version in parallel_versions:
            key = (version, nthreads)
            if key in means and means[key] < best_time:
                best_time = means[key]
                best_version = version
                best_speedup = speedups.get(key, 0)
        if best_version:
            best_lines.append(f"{nthreads:>8} {best_version:<14} {best_time:>10.4f} {best_speedup:>10.3f}")
    best_lines.append("")

    speedup_table = "\n".join(speedup_lines)
    best_table = "\n".join(best_lines)

    return speedups, parallel_versions, thread_counts, speedup_table, best_table

def plot_speedups(speedups, versions, thread_counts):
    fig, ax = plt.subplots(figsize=(8, 6))

    max_threads = max(thread_counts)

    ax.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.4, label='Ideal linear speedup')

    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = plt.cm.tab10.colors

    for idx, version in enumerate(versions):
        threads = []
        spds = []
        for nthreads in thread_counts:
            key = (version, nthreads)
            if key in speedups:
                threads.append(nthreads)
                spds.append(speedups[key])
        if threads:
            ax.plot(threads, spds,
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    linewidth=2, markersize=8,
                    label=version)

    ax.set_xlabel('Number of threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Speedup vs Number of Threads', fontsize=14)
    ax.set_xticks(thread_counts)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max_threads + 1)
    ax.set_ylim(0, max_threads + 1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'speedup_plot.png'), dpi=150)
    print(f"Plot saved to {OUTDIR}/speedup_plot.png")
    plt.show()

def save_and_print(text, filename):
    print(text)
    filepath = os.path.join(OUTDIR, filename)
    with open(filepath, "w") as f:
        f.write(text + "\n")
    print(f"  -> Saved to {filepath}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze timing results")
    parser.add_argument("--baseline", default="seq",
                        help="'seq' = sequential baseline (default), 'self' = each version's own 1-thread run")
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    groups = read_timings("timings.txt")

    print("\n--- Timing Results ---\n")
    timing_table = build_timing_table(groups)
    save_and_print(timing_table, "timing_table.txt")

    print(f"--- Speedup Analysis (baseline: {args.baseline}) ---\n")
    speedups, versions, thread_counts, speedup_table, best_table = compute_speedups(groups, baseline_mode=args.baseline)
    save_and_print(speedup_table, "speedup_table.txt")

    print("--- Best Version per Thread Count ---\n")
    save_and_print(best_table, "best_version_table.txt")

    plot_speedups(speedups, versions, thread_counts)
