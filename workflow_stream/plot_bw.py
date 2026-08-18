#!/usr/bin/env python3
"""
Scan benchmark logs and plot bandwidth curves.

Log filename patterns understood:
    mpi_test_n{NODES}_N{RANKS}_buff{BYTES}.log
    adios2_{engine}_n{NODES}_N{RANKS}_buff{BYTES}.log
    adios2_{engine}_{sst_mode}_{data_plane}_n{NODES}_N{RANKS}_buff{BYTES}.log

Metric lines extracted from the log body:
    per-rank    -> "Avg per-pair bandwidth (from recv time): X GB/s"   (mpi)
                   "Avg per-rank bandwidth (from get time): X GB/s"    (adios2)
    aggregate   -> "Aggregate bandwidth (from wall-clock barriers): X GB/s"

Examples:
    plot_bw.py --nodes 2 --nic-bw 25
        -> single subplot: BW vs data size for 2 nodes, all impls, NIC line at 25 GB/s
    plot_bw.py --nodes all --data-size 1073741824 --nic-bw 25
        -> single subplot: BW vs nodes for 1 GB messages, all impls
    plot_bw.py --nodes 2 --ranks-per-node 1,8,12 --nic-bw 25,200,200
        -> three subplots (1 rpn, 8 rpn, 12 rpn), NIC line per panel
           (e.g. 25 GB/s for 1 rank saturating 1 NIC, 200 GB/s for 8+ ranks
            saturating all 8 Aurora NICs)
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# ---------- log parsing ----------

# Current MPI pattern: N is ranks-per-node.
# Legacy (mpi_test_...) had N as total ranks; still parsed for backward compatibility.
MPI_FNAME = re.compile(r"^mpi_n(?P<nodes>\d+)_N(?P<rpn>\d+)_buff(?P<bytes>\d+)\.log$")
MPI_FNAME_LEGACY = re.compile(r"^mpi_test_n(?P<nodes>\d+)_N(?P<ranks>\d+)_buff(?P<bytes>\d+)\.log$")
# Current adios2 pattern: adios[2]_<engine>_<sst_mode>_<data_plane>_<io_mode>_n{NODES}_N{RPN}_buff{B}.log
# N is now ranks-per-node (same convention as MPI). sst_mode/data_plane always
# present; io_mode required.
ADIOS2_FNAME = re.compile(
    r"^adios2?_"
    r"(?P<engine>[a-z0-9]+)"
    r"_(?P<sst_mode>sync|async)"
    r"_(?P<data_plane>[A-Za-z0-9]+)"
    r"_(?P<io_mode>[a-z0-9]+)"
    r"_n(?P<nodes>\d+)_N(?P<rpn>\d+)_buff(?P<bytes>\d+)\.log$"
)
# Experiment-directory pattern used by SmartSim and Dragon runs. The metadata
# lives in the directory name; producer.out / consumer.out inside hold the
# per-side metrics. Current format encodes both component-node count and DB
# node count: 'n<COMPONENT_NODES>d<DB_NODES>'. Legacy format has just 'n<TOTAL>'.
EXPDIR_NAME = re.compile(
    r"^(?P<framework>ssim|dragon)_(?P<deployment>[a-z]+)"
    r"_n(?P<nodes>\d+)(?:d(?P<db_nodes>\d+))?"
    r"_N(?P<rpn>\d+)_buff(?P<bytes>\d+)$"
)

PER_RANK_LINE = re.compile(
    r"Avg per-(?:pair|rank) bandwidth \(from (?:recv|get) time\):\s*([0-9.eE+-]+)\s*GB/s"
)
AGGREGATE_WALL_LINE = re.compile(
    r"Aggregate bandwidth \(from wall-clock barriers\):\s*([0-9.eE+-]+)\s*GB/s"
)
AGGREGATE_SUM_LINE = re.compile(
    r"Aggregate bandwidth \(sum of per-(?:pair|rank) rates\):\s*([0-9.eE+-]+)\s*GB/s"
)
# Consumer-only fallback: no wall-clock barrier version, uses max-get-time form
AGGREGATE_MAX_LINE = re.compile(
    r"Aggregate bandwidth \(from max (?:recv|get) time\):\s*([0-9.eE+-]+)\s*GB/s"
)


def parse_filename(name):
    """Return metadata dict, or None if the filename is not recognized."""
    m = MPI_FNAME.match(name)
    if m:
        nodes = int(m.group("nodes"))
        rpn = int(m.group("rpn"))
        nbytes = int(m.group("bytes"))
        return {
            "impl": "mpi",
            "nodes": nodes,
            "ranks": nodes * rpn,
            "ranks_per_node": rpn,
            "bytes_per_rank": nbytes,
        }
    m = MPI_FNAME_LEGACY.match(name)
    if m:
        nodes = int(m.group("nodes"))
        ranks = int(m.group("ranks"))
        nbytes = int(m.group("bytes"))
        return {
            "impl": "mpi",
            "nodes": nodes,
            "ranks": ranks,
            "ranks_per_node": ranks // nodes,
            "bytes_per_rank": nbytes,
        }
    m = ADIOS2_FNAME.match(name)
    if m:
        engine = m.group("engine")
        # For SST, include mode + data_plane; for BP5, those fields are populated by
        # the submit script but semantically meaningless -- collapse to just "adios2_bp5".
        if engine == "bp5":
            base_impl = "adios2_bp5"
        else:
            base_impl = f"adios2_{engine}_{m.group('sst_mode')}_{m.group('data_plane')}".lower()
        nodes = int(m.group("nodes"))
        rpn = int(m.group("rpn"))
        return {
            "impl": base_impl,      # base name; scan_logs appends _producer / _consumer
            "engine": engine,
            "nodes": nodes,
            "ranks": nodes * rpn,
            "ranks_per_node": rpn,
            "bytes_per_rank": int(m.group("bytes")),
        }
    return None


def parse_body(path, start_marker=None, stop_marker=None):
    """Extract per-rank and aggregate BW values from a log file.

    If ``start_marker`` is set, lines before the first occurrence are ignored.
    If ``stop_marker`` is set, parsing stops when a line contains it. Used to
    isolate a single component's summary in MPMD logs that concatenate multiple
    summaries (e.g. ADIOS2's producer + consumer stdout).
    """
    per_rank = None
    aggregate_wall = None
    aggregate_sum = None
    aggregate_max = None
    active = start_marker is None
    try:
        with open(path) as f:
            for line in f:
                if not active:
                    if start_marker in line:
                        active = True
                    continue
                if stop_marker is not None and stop_marker in line:
                    break
                m = PER_RANK_LINE.search(line)
                if m:
                    per_rank = float(m.group(1))
                    continue
                m = AGGREGATE_WALL_LINE.search(line)
                if m:
                    aggregate_wall = float(m.group(1))
                    continue
                m = AGGREGATE_SUM_LINE.search(line)
                if m:
                    aggregate_sum = float(m.group(1))
                    continue
                m = AGGREGATE_MAX_LINE.search(line)
                if m:
                    aggregate_max = float(m.group(1))
    except OSError as e:
        print(f"WARN: could not read {path}: {e}", file=sys.stderr)
    return per_rank, aggregate_wall, aggregate_sum, aggregate_max


def scan_expdir(exp_dir):
    """Parse a SmartSim/Dragon experiment directory into up to two records.

    Metadata comes from the directory name; producer.out / consumer.out inside
    (searched recursively so SmartSim's <exp>/{producer,consumer}/*.out also
    matches) provide the per-side metrics. Producer and consumer become separate
    records with impls suffixed "_producer" and "_consumer", so both curves
    appear on the plot.
    """
    m = EXPDIR_NAME.match(exp_dir.name)
    if not m:
        return []
    framework = m.group("framework")
    deployment = m.group("deployment")
    nodes = int(m.group("nodes"))
    rpn = int(m.group("rpn"))
    nbytes = int(m.group("bytes"))
    base_impl = f"{framework}_{deployment}"
    common = {
        "nodes": nodes,
        "ranks": nodes * rpn,
        "ranks_per_node": rpn,
        "bytes_per_rank": nbytes,
    }

    records = []
    # Producer side: per-rank from put time, wall-clock aggregate, sum-of-rates
    for prod in list(exp_dir.rglob("producer.out")) + list(exp_dir.rglob("producer/stdout")):
        per_rank, agg_wall, agg_sum, agg_max = parse_body(prod)
        if all(v is None for v in (per_rank, agg_wall, agg_sum, agg_max)):
            continue
        records.append({
            **common,
            "impl": f"{base_impl}_producer",
            "per_rank_bw": per_rank,
            "aggregate_wall_bw": agg_wall,
            "aggregate_sum_bw": agg_sum,
            "aggregate_max_bw": agg_max,
            "path": str(prod),
        })
        break

    # Consumer side: per-rank from get time, max-time aggregate, sum-of-rates
    for cons in list(exp_dir.rglob("consumer.out")) + list(exp_dir.rglob("consumer/stdout")):
        per_rank, agg_wall, agg_sum, agg_max = parse_body(cons)
        if all(v is None for v in (per_rank, agg_wall, agg_sum, agg_max)):
            continue
        records.append({
            **common,
            "impl": f"{base_impl}_consumer",
            "per_rank_bw": per_rank,
            "aggregate_wall_bw": agg_wall,
            "aggregate_sum_bw": agg_sum,
            "aggregate_max_bw": agg_max,
            "path": str(cons),
        })
        break

    return records


def scan_logs(log_dir):
    """Walk log_dir recursively; return list of records.

    Two source types:
      1. Single-file logs (MPI, ADIOS2): `*.log` whose filename encodes metadata.
      2. Experiment directories (SmartSim, Dragon): directory whose name encodes
         metadata, containing producer.out / consumer.out inside.
    """
    records = []
    root = Path(log_dir).resolve()

    # Single-file logs (MPI, ADIOS2). The ADIOS2 MPMD log contains BOTH the
    # producer and consumer summaries; we split them the same way we split
    # SmartSim/Dragon exp dirs -- one record per side, with the impl suffix
    # "_producer" or "_consumer".
    #
    # ADIOS2 emits BP5 producer numbers for real; for SST the producer numbers
    # are all zeros (bytes actually move at consumer Get time), so we skip that
    # record entirely.
    for path in root.rglob("*.log"):
        meta = parse_filename(path.name)
        if not meta:
            continue

        if meta["impl"].startswith("adios2"):
            base_impl = meta["impl"]
            engine = meta.pop("engine")

            emit_producer = (engine == "bp5")
            emit_consumer = True

            if emit_producer:
                per_rank, agg_wall, agg_sum, agg_max = parse_body(
                    path,
                    start_marker="=== Producer Performance Summary ===",
                    stop_marker="=== Consumer Performance Summary ===",
                )
                if not all(v is None for v in (per_rank, agg_wall, agg_sum, agg_max)):
                    rec = dict(meta)
                    rec["impl"] = f"{base_impl}_producer"
                    rec["per_rank_bw"] = per_rank
                    rec["aggregate_wall_bw"] = agg_wall
                    rec["aggregate_sum_bw"] = agg_sum
                    rec["aggregate_max_bw"] = agg_max
                    rec["path"] = str(path)
                    records.append(rec)

            if emit_consumer:
                per_rank, agg_wall, agg_sum, agg_max = parse_body(
                    path,
                    start_marker="=== Consumer Performance Summary ===",
                    stop_marker="=== Producer Performance Summary ===",
                )
                if not all(v is None for v in (per_rank, agg_wall, agg_sum, agg_max)):
                    rec = dict(meta)
                    rec["impl"] = f"{base_impl}_consumer"
                    rec["per_rank_bw"] = per_rank
                    rec["aggregate_wall_bw"] = agg_wall
                    rec["aggregate_sum_bw"] = agg_sum
                    rec["aggregate_max_bw"] = agg_max
                    rec["path"] = str(path)
                    records.append(rec)
        else:
            per_rank, agg_wall, agg_sum, agg_max = parse_body(path)
            if all(v is None for v in (per_rank, agg_wall, agg_sum, agg_max)):
                continue
            meta["per_rank_bw"] = per_rank
            meta["aggregate_wall_bw"] = agg_wall
            meta["aggregate_sum_bw"] = agg_sum
            meta["aggregate_max_bw"] = agg_max
            meta["path"] = str(path)
            records.append(meta)

    # Experiment directories (SmartSim, Dragon) -- one record per side
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        records.extend(scan_expdir(d))

    return records


# ---------- filtering ----------

def parse_filter(arg, all_values):
    """'all' -> (sorted(all_values), False); '1,2,3' -> ([1,2,3], True); None -> (None, False).

    Second element is `explicit`: True if the user typed a value list, False if
    'all' (or None). Used by the faceting logic to distinguish "user wants a
    subplot per value" from "user didn't restrict".
    """
    if arg is None:
        return None, False
    if arg == "all":
        return sorted(all_values), False
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [int(p) if p.lstrip("-").isdigit() else p for p in parts], True


def parse_impl_filter(arg, all_impls):
    if arg is None:
        return None, False
    if arg == "all":
        return sorted(all_impls), False
    return [p.strip() for p in arg.split(",") if p.strip()], True


def apply_filter(records, key, values):
    if values is None:
        return records
    values = set(values)
    return [r for r in records if r[key] in values]


# ---------- plotting ----------

def label_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:g} {unit}"
        n /= 1024
    return f"{n:g} PB"


METRIC_COLUMN = {
    "per_rank": "per_rank_bw",
    "aggregate_wall": "aggregate_wall_bw",
    "aggregate_sum": "aggregate_sum_bw",
    "aggregate_max": "aggregate_max_bw",
}

METRIC_YLABEL = {
    "per_rank": "Per-Rank Bandwidth (GB/s)",
    "aggregate_wall": "Aggregate BW, wall-clock (GB/s)",
    "aggregate_sum": "Aggregate BW, sum of per-rank rates (GB/s)",
    "aggregate_max": "Aggregate BW, max-time (GB/s)",
}


def bw_column(metric):
    return METRIC_COLUMN[metric]


def bw_ylabel(metric):
    return METRIC_YLABEL[metric]


def plot_series(ax, records, x_key, metric, impls, impl_style, nic_bw):
    """Draw one axes: lines per impl, x = x_key, y = BW.

    impl_style maps impl -> (color, linestyle) so every subplot uses the same
    color for the same implementation. When x_key == "bytes_per_rank" the values
    are converted to GB (bytes/1e9) so the units match the y-axis (GB/s).
    """
    col = bw_column(metric)
    x_scale = 1e9 if x_key == "bytes_per_rank" else 1
    for impl in sorted(impls):
        pts = sorted(
            [(r[x_key], r[col]) for r in records if r["impl"] == impl and r[col] is not None],
            key=lambda p: p[0],
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        xs = tuple(x / x_scale for x in xs)
        color, linestyle = impl_style[impl]
        ax.plot(xs, ys, marker="o", color=color, linestyle=linestyle, label=impl)

    if nic_bw is not None:
        ax.axhline(nic_bw, color="k", linewidth=1, label=f"NIC BW ({nic_bw:g} GB/s)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def build_impl_style(impls):
    """Assign a stable (color, linestyle) to each impl so subplots stay consistent."""
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ])
    styles = ["-", "--", ":", "-."]
    result = {}
    for i, impl in enumerate(sorted(impls)):
        result[impl] = (palette[i % len(palette)], styles[i % len(styles)])
    return result


# Facet axis chosen in this priority order: whatever the user gave a multi-item list for.
# Falls back to any dimension that happens to have multiple unique values in the data.
FACET_PRIORITY = [
    ("nodes", "nodes", "Nodes"),
    ("ranks_per_node", "ranks_per_node", "Ranks per Node"),
    ("bytes_per_rank", "bytes_per_rank", "Data Size"),
    ("impl", "impl", "Implementation"),
]


def choose_facet(filters, records, x_key):
    """Return (record_key, values) to facet on, or (None, [None]) for a single panel.

    Rule: use the first filter (in FACET_PRIORITY order) that the user EXPLICITLY listed
    with >=2 values. 'all' (or an unset filter) does not trigger auto-faceting -- the
    user only gets a subplot per value when they typed a comma-separated list.
    Impls always appear as separate curves within a panel. The x-axis is never a facet.
    """
    for _, key, _ in FACET_PRIORITY:
        if key == x_key:
            continue
        entry = filters.get(key)
        if entry is None:
            continue
        values, explicit = entry
        if explicit and values is not None and len(values) > 1:
            return key, sorted(values)
    return None, [None]


def panel_title(key, value):
    if key == "nodes":
        return f"{value} Node{'s' if value != 1 else ''}"
    if key == "ranks_per_node":
        return f"{value} Rank{'s' if value != 1 else ''} per Node"
    if key == "bytes_per_rank":
        return f"{label_bytes(value)} per rank"
    if key == "impl":
        return str(value)
    return ""


def make_plots(records, args, filters):
    if not records:
        print("No records to plot after filtering.", file=sys.stderr)
        return 1

    # X-axis choice: if the user explicitly pinned a single data-size, plot BW vs nodes;
    # otherwise BW vs data size.
    size_entry = filters.get("bytes_per_rank")
    size_values, size_explicit = size_entry if size_entry else (None, False)
    x_key = "nodes" if (size_explicit and size_values is not None and len(size_values) == 1) \
                    else "bytes_per_rank"

    facet_key, facet_values = choose_facet(filters, records, x_key)

    # Build the stable impl -> (color, linestyle) map from the full set of impls
    # so all subplots use the same color for the same implementation.
    impls = sorted({r["impl"] for r in records})
    impl_style = build_impl_style(impls)

    if facet_key is None:
        panels = [(None, records)]
    else:
        panels = [(v, [r for r in records if r[facet_key] == v]) for v in facet_values]

    # Validate --nic-bw length against the facet.
    n_panels = len(panels)
    if len(args.nic_bw) == 1:
        panel_nics = args.nic_bw * n_panels
    elif len(args.nic_bw) == n_panels:
        panel_nics = args.nic_bw
    else:
        if facet_key == "ranks_per_node":
            print(f"ERROR: --nic-bw got {len(args.nic_bw)} values but faceting by "
                  f"ranks_per_node produces {n_panels} panels for values "
                  f"{facet_values}. Pass one nic-bw per rpn value, or a single "
                  f"value to reuse across all panels.", file=sys.stderr)
        else:
            print(f"ERROR: --nic-bw got {len(args.nic_bw)} values but plot has "
                  f"{n_panels} panel(s). Pass one value, or one per panel.",
                  file=sys.stderr)
        return 1

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)

    for ax, (facet_val, panel_records), panel_nic in zip(axes[0], panels, panel_nics):
        plot_series(ax, panel_records, x_key, args.metric, impls, impl_style, panel_nic)
        if facet_key is not None:
            ax.set_title(panel_title(facet_key, facet_val))
        if x_key == "bytes_per_rank":
            ax.set_xlabel("Data Size per Rank (GB)")
        else:
            ax.set_xlabel("Number of Nodes")
        ax.set_ylabel(bw_ylabel(args.metric))

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", default=".", help="directory to scan for *.log (recursive). Default: cwd")
    p.add_argument("--nodes", default="all", help="comma-separated node counts, or 'all'")
    p.add_argument("--ranks-per-node", default="all", help="comma-separated ranks-per-node, or 'all'")
    p.add_argument("--data-size", default="all", help="comma-separated bytes-per-rank values, or 'all'")
    p.add_argument("--impl", default="all", help="comma-separated implementations (mpi, adios2_bp5, adios2_sst_sync_rdma, ...), or 'all'")
    p.add_argument("--metric",
                   choices=["per_rank", "aggregate_wall", "aggregate_sum", "aggregate_max"],
                   default="per_rank",
                   help="which BW to plot: per_rank | aggregate_wall (bytes/wall-clock, producer-side) "
                        "| aggregate_sum (sum of per-rank rates) | aggregate_max (bytes/max-time, consumer-side)")
    p.add_argument("--nic-bw", type=str, required=True,
                   help="Comma-separated NIC BW ceiling(s) in GB/s to draw as a horizontal "
                        "reference line, one per subplot. Pass a single value to reuse across "
                        "all panels; when faceting by ranks_per_node, pass one value per rpn "
                        "(order-matched to the sorted rpn list). Give the aggregate value you "
                        "actually expect for that rpn (i.e. account for however many NICs that "
                        "rpn saturates), not the per-NIC ceiling.")
    p.add_argument("--output", "-o", default="bw_plot.png", help="output PNG path")
    args = p.parse_args()

    # Parse comma-separated --nic-bw into a list of floats
    try:
        args.nic_bw = [float(x.strip()) for x in args.nic_bw.split(",") if x.strip()]
    except ValueError as e:
        print(f"ERROR: --nic-bw must be a comma-separated list of numbers ({e})", file=sys.stderr)
        return 1
    if not args.nic_bw:
        print("ERROR: --nic-bw must have at least one value", file=sys.stderr)
        return 1

    records = scan_logs(args.log_dir)
    if not records:
        print(f"No parseable logs found under {args.log_dir}", file=sys.stderr)
        return 1
    print(f"Scanned {len(records)} logs from {args.log_dir}", file=sys.stderr)

    # Build the universe of values from all records BEFORE filtering,
    # so 'all' means 'everything present on disk'.
    all_nodes = {r["nodes"] for r in records}
    all_rpns = {r["ranks_per_node"] for r in records}
    all_sizes = {r["bytes_per_rank"] for r in records}
    all_impls = {r["impl"] for r in records}

    node_filter, node_explicit = parse_filter(args.nodes, all_nodes)
    rpn_filter, rpn_explicit = parse_filter(args.ranks_per_node, all_rpns)
    size_filter, size_explicit = parse_filter(args.data_size, all_sizes)
    impl_filter, impl_explicit = parse_impl_filter(args.impl, all_impls)

    records = apply_filter(records, "nodes", node_filter)
    records = apply_filter(records, "ranks_per_node", rpn_filter)
    records = apply_filter(records, "bytes_per_rank", size_filter)
    records = apply_filter(records, "impl", impl_filter)

    filters = {
        "nodes": (node_filter, node_explicit),
        "ranks_per_node": (rpn_filter, rpn_explicit),
        "bytes_per_rank": (size_filter, size_explicit),
        "impl": (impl_filter, impl_explicit),
    }
    return make_plots(records, args, filters)


if __name__ == "__main__":
    sys.exit(main())
