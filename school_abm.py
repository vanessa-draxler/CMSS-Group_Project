from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mesa
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Population
N_GRADES = 5
N_CLASSES_PER_GRADE = 2
N_STUDENTS_PER_CLASS = 15

# Simulation time
N_DAYS = 10
N_RUNS = 30
BASE_SEED = 42

# Extroversion
EXTROVERSION_MEAN = 0.5
EXTROVERSION_STD = 0.15

# Susceptibility
SUSCEPTIBILITY_MEAN = 0.4
SUSCEPTIBILITY_STD = 0.15

# Friendship graph (edge probability by relationship type)
P_EDGE_SAME_CLASS = 0.30
P_EDGE_SAME_GRADE = 0.05
P_EDGE_CROSS_GRADE = 0.005

# Friendship weight distributions
W_INIT_SAME_CLASS_MEAN = 0.6
W_INIT_SAME_CLASS_STD = 0.1
W_INIT_SAME_GRADE_MEAN = 0.2
W_INIT_SAME_GRADE_STD = 0.1
W_INIT_CROSS_GRADE_MEAN = 0.05
W_INIT_CROSS_GRADE_STD = 0.1

# Max edges per node
MAX_FRIENDS = 15

# weight above which agents are "friends"
FRIENDSHIP_THRESHOLD = 5.0

# Friendship weight
LAMBDA_DECAY = 0.98
WEIGHT_INCREMENT = 1.0

# Interaction probability formula
ALPHA = 0.01
H_CLASS_SAME = 1.0  # homophily: same class
H_CLASS_DIFF = 0.2  # homophily: different class
H_GRADE_SAME = 1.0  # homophily: same grade
H_GRADE_DIFF = 0.5  # homophily: different grade

# Trust multiplier
TRUST_FRIEND = 1.0
TRUST_SAME_CLASS = 0.6
TRUST_SAME_GRADE = 0.3
TRUST_CROSS_GRADE = 0.1

# Data collection
DATA_COLLECTION_INTERVAL_MINUTES = 60  # collect every N timesteps
DISTRIBUTION_LOWER_QUANTILE = 0.10
DISTRIBUTION_UPPER_QUANTILE = 0.90

# School day period schedule
SCHOOL_DAY = [
    {"name": "lesson", "duration_min": 50, "interaction_multiplier": 0.2},
    {"name": "short_break", "duration_min": 10, "interaction_multiplier": 1.0},
    {"name": "lesson", "duration_min": 50, "interaction_multiplier": 0.2},
    {"name": "short_break", "duration_min": 10, "interaction_multiplier": 1.0},
    {"name": "lesson", "duration_min": 50, "interaction_multiplier": 0.2},
    {"name": "short_break", "duration_min": 10, "interaction_multiplier": 1.0},
    {"name": "lesson", "duration_min": 50, "interaction_multiplier": 0.2},
    {"name": "lunch", "duration_min": 60, "interaction_multiplier": 1.5},
    {"name": "lesson", "duration_min": 50, "interaction_multiplier": 0.2},
    {"name": "short_break", "duration_min": 10, "interaction_multiplier": 1.0},
    {"name": "lesson", "duration_min": 50, "interaction_multiplier": 0.2},
    {"name": "short_break", "duration_min": 10, "interaction_multiplier": 1.0},
    {"name": "lesson", "duration_min": 50, "interaction_multiplier": 0.2},
]

STATE_UNAWARE = "Unaware"
STATE_EXPOSED = "Exposed"
STATE_BELIEVER = "Believer"
STATE_SHARER = "Sharer"
STATE_ORDER = [
    STATE_UNAWARE,
    STATE_EXPOSED,
    STATE_BELIEVER,
    STATE_SHARER,
]

TOTAL_STUDENTS = N_GRADES * N_CLASSES_PER_GRADE * N_STUDENTS_PER_CLASS
TOTAL_DAY_MINUTES = sum(period["duration_min"] for period in SCHOOL_DAY)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = SCRIPT_DIR / "results.csv"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "results_summary.csv"
DEFAULT_PLOT_PATH = SCRIPT_DIR / "results.png"
DEFAULT_NETWORK_PLOT_PATH = SCRIPT_DIR / "network_spread.png"
MIN_NETWORK_SNAPSHOTS = 2
MAX_NETWORK_SNAPSHOTS = 5
STATE_ABBREVIATIONS = {
    STATE_UNAWARE: "U",
    STATE_EXPOSED: "E",
    STATE_BELIEVER: "B",
    STATE_SHARER: "S",
}
STATE_COLORS = {
    STATE_UNAWARE: "#d9d9d9",
    STATE_EXPOSED: "#fdbf6f",
    STATE_BELIEVER: "#e31a1c",
    STATE_SHARER: "#6a3d9a",
}
GRADE_MARKERS = ["o", "s", "^", "D", "P"]


class StudentAgent(mesa.Agent):
    def __init__(
        self,
        model: mesa.Model,
        grade: int,
        class_id: int,
    ) -> None:
        super().__init__(model)
        self.grade = grade
        self.class_id = class_id
        self.extroversion = float(
            np.clip(np.random.normal(EXTROVERSION_MEAN, EXTROVERSION_STD), 0.0, 1.0)
        )
        self.susceptibility = float(
            np.clip(np.random.normal(SUSCEPTIBILITY_MEAN, SUSCEPTIBILITY_STD), 0.0, 1.0)
        )
        self.state = STATE_UNAWARE
        self.exposure_count = 0

    def set_state(self, state: str) -> None:
        logger.debug(
            f"Agent {self.unique_id} transitioned from {self.state} to {state}"
        )
        self.state = state


def _clip_weight(value: float) -> float:
    return float(np.clip(value, 0.01, 1.0))


def initialize_network(agents: Iterable[StudentAgent]) -> nx.Graph:
    graph = nx.Graph()
    agent_list = list(agents)
    graph.add_nodes_from(agent.unique_id for agent in agent_list)

    for idx_i, agent_i in enumerate(agent_list):
        for agent_j in agent_list[idx_i + 1 :]:
            if agent_i.grade == agent_j.grade and agent_i.class_id == agent_j.class_id:
                p_edge = P_EDGE_SAME_CLASS
                weight = _clip_weight(
                    np.random.normal(W_INIT_SAME_CLASS_MEAN, W_INIT_SAME_CLASS_STD)
                )
            elif agent_i.grade == agent_j.grade:
                p_edge = P_EDGE_SAME_GRADE
                weight = _clip_weight(
                    np.random.normal(W_INIT_SAME_GRADE_MEAN, W_INIT_SAME_GRADE_STD)
                )
            else:
                p_edge = P_EDGE_CROSS_GRADE
                weight = _clip_weight(
                    np.random.normal(W_INIT_CROSS_GRADE_MEAN, W_INIT_CROSS_GRADE_STD)
                )

            if np.random.random() < p_edge:
                if (
                    graph.degree(agent_i.unique_id) < MAX_FRIENDS
                    and graph.degree(agent_j.unique_id) < MAX_FRIENDS
                ):
                    graph.add_edge(agent_i.unique_id, agent_j.unique_id, weight=weight)

    return graph


def sample_interaction_duration(period_name: str) -> int:
    _ = period_name  # period_name reserved for future refinements
    return 1 if np.random.random() < 0.88 else 5


def compute_interaction_probability(
    agent_i: StudentAgent,
    agent_j: StudentAgent,
    w_ij: float,
    period_multiplier: float,
) -> float:
    h_class = (
        H_CLASS_SAME
        if agent_i.class_id == agent_j.class_id and agent_i.grade == agent_j.grade
        else H_CLASS_DIFF
    )
    h_grade = H_GRADE_SAME if agent_i.grade == agent_j.grade else H_GRADE_DIFF
    p = (
        ALPHA
        * h_class
        * h_grade
        * w_ij
        * agent_i.extroversion
        * agent_j.extroversion
        * period_multiplier
    )
    return float(min(1.0, p))


def weighted_sample_without_replacement(
    rng, candidates: list[StudentAgent], weights: list[float], sample_size: int
) -> list[StudentAgent]:
    selected: list[StudentAgent] = []
    remaining_candidates = candidates.copy()
    remaining_weights = weights.copy()

    for _ in range(sample_size):
        total_weight = sum(remaining_weights)
        if total_weight <= 0:
            break

        threshold = rng.random() * total_weight
        cumulative = 0.0
        for idx, weight in enumerate(remaining_weights):
            cumulative += weight
            if cumulative >= threshold:
                selected.append(remaining_candidates.pop(idx))
                remaining_weights.pop(idx)
                break

    return selected


def compute_trust(
    agent_i: StudentAgent, agent_j: StudentAgent, network: nx.Graph
) -> float:
    if network.has_edge(agent_i.unique_id, agent_j.unique_id):
        if (
            network[agent_i.unique_id][agent_j.unique_id]["weight"]
            >= FRIENDSHIP_THRESHOLD
        ):
            return TRUST_FRIEND

    if agent_i.grade == agent_j.grade and agent_i.class_id == agent_j.class_id:
        return TRUST_SAME_CLASS
    if agent_i.grade == agent_j.grade:
        return TRUST_SAME_GRADE
    return TRUST_CROSS_GRADE


class SchoolModel(mesa.Model):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(rng=seed)

        grades: list[int] = []
        classes: list[int] = []

        for grade in range(N_GRADES):
            for class_id in range(N_CLASSES_PER_GRADE):
                grades.extend([grade] * N_STUDENTS_PER_CLASS)
                classes.extend([class_id] * N_STUDENTS_PER_CLASS)

        self.students = StudentAgent.create_agents(
            self,
            TOTAL_STUDENTS,
            grade=grades,
            class_id=classes,
        )

        initial_sharer = self.random.choice(list(self.students))
        initial_sharer.set_state(STATE_SHARER)

        self.agents_by_id = {agent.unique_id: agent for agent in self.students}
        self.network = initialize_network(self.students)
        self.current_minute = 0
        self.school_day = SCHOOL_DAY
        self.total_day_minutes = TOTAL_DAY_MINUTES

        self.datacollector = mesa.DataCollector(
            model_reporters={
                state: (lambda m, s=state: m.count_state(s)) for state in STATE_ORDER
            }
        )

    def count_state(self, state: str) -> int:
        return sum(1 for agent in self.students if agent.state == state)

    def _current_period(self) -> dict:
        minute_in_day = self.current_minute % self.total_day_minutes
        elapsed = 0
        for period in self.school_day:
            elapsed += period["duration_min"]
            if minute_in_day < elapsed:
                return period
        return self.school_day[-1]

    def step(self) -> None:
        period = self._current_period()
        period_multiplier = period["interaction_multiplier"]
        period_name = period["name"]
        agents_snapshot = list(self.students)

        for agent_i in agents_snapshot:
            n_interactions = max(1, round(agent_i.extroversion * 5))
            candidates: list[StudentAgent] = []
            weights: list[float] = []

            for neighbor_id in self.network.neighbors(agent_i.unique_id):
                agent_j = self.agents_by_id[neighbor_id]
                w_ij = self.network[agent_i.unique_id][agent_j.unique_id]["weight"]
                p_ij = compute_interaction_probability(
                    agent_i, agent_j, w_ij, period_multiplier
                )
                if p_ij > 0:
                    candidates.append(agent_j)
                    weights.append(p_ij)

            if not candidates:
                continue

            sample_size = min(n_interactions, len(candidates))
            selected_agents = weighted_sample_without_replacement(
                self.random, candidates, weights, sample_size
            )

            for agent_j in selected_agents:
                duration = sample_interaction_duration(period_name)

                if agent_i.state == STATE_SHARER and agent_j.state in {
                    STATE_UNAWARE,
                    STATE_EXPOSED,
                }:
                    trust = compute_trust(agent_i, agent_j, self.network)
                    agent_j.exposure_count += 1
                    if agent_j.state == STATE_UNAWARE:
                        agent_j.set_state(STATE_EXPOSED)
                    elif agent_j.state == STATE_EXPOSED:
                        p_belief = (
                            1.0
                            - (1.0 - agent_j.susceptibility * trust)
                            ** agent_j.exposure_count
                        )
                        if np.random.random() < p_belief:
                            if np.random.random() < agent_j.extroversion:
                                agent_j.set_state(STATE_SHARER)
                            else:
                                agent_j.set_state(STATE_BELIEVER)

                    if self.network.has_edge(agent_i.unique_id, agent_j.unique_id):
                        self.network[agent_i.unique_id][agent_j.unique_id][
                            "weight"
                        ] += WEIGHT_INCREMENT * duration
                    elif (
                        self.network.degree(agent_i.unique_id) < MAX_FRIENDS
                        and self.network.degree(agent_j.unique_id) < MAX_FRIENDS
                    ):
                        self.network.add_edge(
                            agent_i.unique_id,
                            agent_j.unique_id,
                            weight=WEIGHT_INCREMENT * duration,
                        )

        if self.current_minute % self.total_day_minutes == self.total_day_minutes - 1:
            for _, _, data in self.network.edges(data=True):
                data["weight"] *= LAMBDA_DECAY

        if self.current_minute % DATA_COLLECTION_INTERVAL_MINUTES == 0:
            self.datacollector.collect(self)

        self.current_minute += 1


def validate_network_snapshot_count(snapshot_count: int) -> None:
    if not MIN_NETWORK_SNAPSHOTS <= snapshot_count <= MAX_NETWORK_SNAPSHOTS:
        raise ValueError(
            "network snapshot count must be between "
            f"{MIN_NETWORK_SNAPSHOTS} and {MAX_NETWORK_SNAPSHOTS}"
        )


def capture_network_snapshot(model: "SchoolModel") -> dict:
    node_data = {
        agent.unique_id: {
            "grade": agent.grade,
            "class_id": agent.class_id,
            "state": agent.state,
        }
        for agent in model.students
    }
    return {
        "minute": model.current_minute,
        "node_data": node_data,
        "graph": model.network.copy(),
    }


def calculate_clustered_positions(
    node_data: dict[int, dict],
) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    grade_spacing_x = 5.0
    class_spacing_y = 4.0
    node_radius = 1.15
    class_center_offset = (N_CLASSES_PER_GRADE - 1) / 2

    for grade in range(N_GRADES):
        for class_id in range(N_CLASSES_PER_GRADE):
            node_ids = sorted(
                node_id
                for node_id, attributes in node_data.items()
                if attributes["grade"] == grade and attributes["class_id"] == class_id
            )
            if not node_ids:
                continue

            cluster_center_x = grade * grade_spacing_x
            cluster_center_y = (class_center_offset - class_id) * class_spacing_y
            angles = np.linspace(0, 2 * np.pi, len(node_ids), endpoint=False)
            for idx, node_id in enumerate(node_ids):
                positions[node_id] = (
                    cluster_center_x + node_radius * np.cos(angles[idx]),
                    cluster_center_y + node_radius * np.sin(angles[idx]),
                )

    return positions


def run_network_snapshot_simulation(
    seed: int | None = BASE_SEED,
    snapshot_count: int = MAX_NETWORK_SNAPSHOTS,
) -> list[dict]:
    validate_network_snapshot_count(snapshot_count)
    if seed is not None:
        np.random.seed(seed)

    model = SchoolModel(seed=seed)
    total_steps = 1000  # Most interesting stuff happens < 1000 steps
    snapshot_minutes = set(np.linspace(0, total_steps, snapshot_count, dtype=int))
    snapshots: list[dict] = []

    if 0 in snapshot_minutes:
        snapshots.append(capture_network_snapshot(model))

    for _ in range(total_steps):
        model.step()
        if model.current_minute in snapshot_minutes:
            snapshots.append(capture_network_snapshot(model))

    return snapshots


def plot_network_snapshots(
    snapshots: list[dict],
    plot_path: Path = DEFAULT_NETWORK_PLOT_PATH,
    show: bool = False,
) -> None:
    if not snapshots:
        raise ValueError("at least one network snapshot is required")

    snapshot_count = len(snapshots)
    n_cols = min(snapshot_count, 3)
    n_rows = int(np.ceil(snapshot_count / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, snapshot in zip(axes_array, snapshots):
        graph = snapshot["graph"]
        node_data = snapshot["node_data"]
        positions = calculate_clustered_positions(node_data)

        widths = [0.2 + 0.15 * graph[u][v].get("weight", 1.0) for u, v in graph.edges]
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            alpha=0.18,
            edge_color="gray",
            width=widths,
        )

        for grade in range(N_GRADES):
            marker = GRADE_MARKERS[grade % len(GRADE_MARKERS)]
            grade_nodes = [
                node_id
                for node_id, attributes in node_data.items()
                if attributes["grade"] == grade
            ]
            node_colors = [
                STATE_COLORS[node_data[node_id]["state"]] for node_id in grade_nodes
            ]
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=grade_nodes,
                node_color=node_colors,
                node_shape=marker,
                node_size=260,
                edgecolors="black",
                linewidths=0.5,
                ax=ax,
            )

        labels = {
            node_id: STATE_ABBREVIATIONS[attributes["state"]]
            for node_id, attributes in node_data.items()
        }
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            font_size=7,
            font_weight="bold",
            ax=ax,
        )

        state_counts = pd.Series(
            [attributes["state"] for attributes in node_data.values()]
        ).value_counts()
        spread_count = int(
            state_counts.get(STATE_EXPOSED, 0)
            + state_counts.get(STATE_BELIEVER, 0)
            + state_counts.get(STATE_SHARER, 0)
        )
        ax.set_title(
            f"Minute {snapshot['minute']:,}\n"
            f"Exposed/Believer/Sharer: {spread_count}/{TOTAL_STUDENTS}"
        )
        ax.set_axis_off()
        ax.margins(0.12)

    for ax in axes_array[len(snapshots) :]:
        ax.set_axis_off()

    grade_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=GRADE_MARKERS[grade % len(GRADE_MARKERS)],
            color="w",
            label=f"Grade {grade + 1}",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=9,
            linestyle="None",
        )
        for grade in range(N_GRADES)
    ]
    state_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{STATE_ABBREVIATIONS[state]} = {state}",
            markerfacecolor=STATE_COLORS[state],
            markeredgecolor="black",
            markersize=8,
            linestyle="None",
        )
        for state in STATE_ORDER
    ]

    fig.legend(
        handles=grade_handles + state_handles,
        loc="lower center",
        ncol=5,
        fontsize="small",
        bbox_to_anchor=(0.5, 0.04),
    )
    fig.suptitle(
        "Network Snapshots of Misinformation Spread\n"
        "Classes are spatial clusters only; "
        "node color/label show state; shape shows grade",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    fig.savefig(plot_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def run_simulation(seed: int | None = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    model = SchoolModel(seed=seed)
    total_steps = N_DAYS * TOTAL_DAY_MINUTES
    for _ in range(total_steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe().reset_index(names="Collection")
    df["Time (minutes)"] = df["Collection"] * DATA_COLLECTION_INTERVAL_MINUTES
    return df


def summarize_runs(results: pd.DataFrame) -> pd.DataFrame:
    aggregations = ["mean", "std", "min", "max"]
    summary = (
        results.groupby("Time (minutes)")[STATE_ORDER].agg(aggregations).reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    quantiles = (
        results.groupby("Time (minutes)")[STATE_ORDER]
        .quantile([DISTRIBUTION_LOWER_QUANTILE, DISTRIBUTION_UPPER_QUANTILE])
        .unstack(level=-1)
        .reset_index()
    )
    quantiles.columns = [
        f"{state}_q{int(q * 100)}" if isinstance(q, float) else state
        for state, q in quantiles.columns
    ]

    return summary.merge(quantiles, on="Time (minutes)")


def run_simulations(
    n_runs: int = N_RUNS,
    base_seed: int | None = BASE_SEED,
    results_path: Path = DEFAULT_RESULTS_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    runs: list[pd.DataFrame] = []

    for run_id in tqdm(range(n_runs)):
        seed = None if base_seed is None else base_seed + run_id
        run_df = run_simulation(seed=seed)
        run_df.insert(0, "Run", run_id + 1)
        if seed is not None:
            run_df.insert(1, "Seed", seed)
        runs.append(run_df)

    results = pd.concat(runs, ignore_index=True)
    summary = summarize_runs(results)

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    return results, summary


def plot_results(
    results: pd.DataFrame,
    plot_path: Path = DEFAULT_PLOT_PATH,
    show: bool = False,
) -> None:
    summary = summarize_runs(results)
    x = summary["Time (minutes)"].to_numpy()

    plt.figure(figsize=(10, 6))
    for state in STATE_ORDER:
        mean = summary[f"{state}_mean"].to_numpy()
        lower = summary[f"{state}_q{int(DISTRIBUTION_LOWER_QUANTILE * 100)}"].to_numpy()
        upper = summary[f"{state}_q{int(DISTRIBUTION_UPPER_QUANTILE * 100)}"].to_numpy()

        line = plt.plot(x, mean, label=f"{state} mean")[0]
        plt.fill_between(
            x,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.18,
            linewidth=0,
            label=f"{state} {int(DISTRIBUTION_LOWER_QUANTILE * 100)}–{int(DISTRIBUTION_UPPER_QUANTILE * 100)}%",
        )

    plt.xlabel("Time (minutes)")
    plt.ylabel("Number of Agents")
    plt.title("Misinformation Spread Over Time Across Runs")
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the school misinformation ABM repeatedly and plot state distributions."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=N_RUNS,
        help=f"number of independent simulation runs (default: {N_RUNS})",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
        help="base random seed; each run uses base_seed + run_id (default: 42)",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="disable deterministic seeding across runs",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="where to write per-run time-series data",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="where to write aggregated distribution statistics",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="where to write the distribution plot",
    )
    parser.add_argument(
        "--network-plot-path",
        type=Path,
        default=DEFAULT_NETWORK_PLOT_PATH,
        help="where to write the network snapshot plot",
    )
    parser.add_argument(
        "--network-snapshots",
        type=int,
        default=MAX_NETWORK_SNAPSHOTS,
        help=(
            "number of uniformly spaced network snapshots to plot "
            f"(default: {MAX_NETWORK_SNAPSHOTS}; "
            f"allowed: {MIN_NETWORK_SNAPSHOTS}-{MAX_NETWORK_SNAPSHOTS})"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the plot window after saving it",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_network_snapshot_count(args.network_snapshots)
    base_seed = None if args.no_seed else args.base_seed
    results_df, _summary_df = run_simulations(
        n_runs=args.runs,
        base_seed=base_seed,
        results_path=args.results_path,
        summary_path=args.summary_path,
    )
    plot_results(results_df, plot_path=args.plot_path, show=args.show)

    network_seed = None if base_seed is None else base_seed
    snapshots = run_network_snapshot_simulation(
        seed=network_seed,
        snapshot_count=args.network_snapshots,
    )
    plot_network_snapshots(
        snapshots,
        plot_path=args.network_plot_path,
        show=args.show,
    )
