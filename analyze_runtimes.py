import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def read_timings(filename):
    data = defaultdict(list)
    with open(filename, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"Skipping line {lineno}: expected 5 columns, got {len(parts)}")
                continue
            try:
                runtime = float(parts[0])
                nparticles = int(parts[1])
                version = parts[2]
                ntimesteps = int(parts[3])
                nthreads = int(parts[4])
            except ValueError as e:
                print(f"Skipping line {lineno}: parse error: {e}")
                continue
            key = (nparticles, version, ntimesteps, nthreads)
            data[key].append(runtime)
    return data


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def build_lines(data):
    """
    Build one line per (version, nthreads) combination.
    Each line has points at different Nparticles values.
    Returns dict: (version, nthreads) -> sorted list of (Nparticles, mean_runtime, std_runtime)
    """
    lines = defaultdict(list)
    for (nparticles, version, ntimesteps, nthreads), runtimes in data.items():
        lines[(version, nthreads)].append((nparticles, mean(runtimes), std(runtimes)))

    for key in lines:
        lines[key].sort(key=lambda x: x[0])

    return lines


def plot_all(lines, output=None, title=None, loglog=False):
    fig, ax = plt.subplots(figsize=(10, 7))

    markers = ['o', 's', '^', 'D', 'v', 'p', 'X', '*']
    colors = plt.cm.tab10.colors

    sorted_keys = sorted(lines.keys(), key=lambda k: (k[0], k[1]))

    for idx, (version, nthreads) in enumerate(sorted_keys):
        points = lines[(version, nthreads)]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        errs = [p[2] for p in points]
        label = f"{version} (T={nthreads})"

        ax.plot(xs, ys,
               marker=markers[idx % len(markers)],
               color=colors[idx % len(colors)],
               linewidth=2, markersize=7,
               label=label)

    # O(N²) reference line scaled to match the data
    all_xs = []
    all_ys = []
    for points in lines.values():
        all_xs.extend(p[0] for p in points)
        all_ys.extend(p[1] for p in points)
    if all_xs and all_ys:
        import numpy as np
        ref_xs = np.array(sorted(set(all_xs)))
        # Scale O(N²) to pass through the median data point
        mid_idx = len(ref_xs) // 2
        mid_x = ref_xs[mid_idx]
        mid_y = np.median(all_ys)
        c = mid_y / (mid_x ** 2)
        ref_ys = c * ref_xs ** 2
        ax.plot(ref_xs, ref_ys, 'k--', alpha=0.4, linewidth=1.5, label='O(N²) reference')

    ax.set_xlabel('Nparticles', fontsize=12)
    ax.set_ylabel('Mean runtime (s)', fontsize=12)
    ax.set_title(title or 'Runtime vs Nparticles', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output}")

    plt.show()


def print_summary_table(lines):
    print("\n" + "=" * 75)
    print(f"{'Version':<12} {'Threads':>8} {'N':>8} {'Mean (s)':>12} {'Std (s)':>12}")
    print("=" * 75)

    for (version, nthreads) in sorted(lines.keys(), key=lambda k: (k[0], k[1])):
        for nparticles, m, s in lines[(version, nthreads)]:
            print(f"{version:<12} {nthreads:>8} {nparticles:>8} {m:>12.6f} {s:>12.6f}")
        print("-" * 75)
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot runtime vs Nparticles, all in one plot")
    parser.add_argument("filename", nargs="?", default="timings_runtimes.txt", help="Input timing file")
    parser.add_argument("--output", default="plots/runtime_vs_N.png", help="Output image path")
    parser.add_argument("--title", default="Runtime vs Nparticles", help="Plot title")
    parser.add_argument("--loglog", action="store_true", help="Use log-log axes")
    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        raise FileNotFoundError(f"Could not find file: {args.filename}")

    data = read_timings(args.filename)
    lines = build_lines(data)

    print_summary_table(lines)
    plot_all(lines, output=args.output, title=args.title, loglog=args.loglog)


if __name__ == "__main__":
    main()
