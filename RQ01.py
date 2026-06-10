from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal
from tqdm import tqdm

import school_abm


OUTPUT_DIR = Path("RQ01_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

N_RUNS = 50
BASE_SEED = 42

AFFECTED_STATES = [
    school_abm.STATE_EXPOSED,
    school_abm.STATE_BELIEVER,
    school_abm.STATE_SHARER,
]

STARTER_LABELS = {
    "random": "Random starter",
    "high_degree": "Highly connected starter",
    "low_degree": "Low-connected starter",
    "high_extroversion": "Highly extroverted starter",
}

STARTER_ORDER = [
    "low_degree",
    "random",
    "high_degree",
    "high_extroversion",
]


def reset_model_with_starter(model, starter_strategy):
    """
    Reset all students to Unaware and choose a new initial Sharer
    according to the selected starter strategy.

    This allows us to test whether the first spreader's individual traits
    and network position affect misinformation diffusion.

    """

    for student in model.students:
        student.state = school_abm.STATE_UNAWARE
        student.exposure_count = 0

    students = list(model.students)

    if starter_strategy == "random":
        starter = model.random.choice(students)

    elif starter_strategy == "high_degree":
        starter = max(
            students,
            key=lambda student: model.network.degree(student.unique_id),
        )

    elif starter_strategy == "low_degree":
        starter = min(
            students,
            key=lambda student: model.network.degree(student.unique_id),
        )

    elif starter_strategy == "high_extroversion":
        starter = max(
            students,
            key=lambda student: student.extroversion,
        )

    else:
        raise ValueError(f"Unknown starter strategy: {starter_strategy}")

    starter.set_state(school_abm.STATE_SHARER)

    return starter


def run_single_experiment(seed, starter_strategy):
    """
    Run one simulation for a specific starter strategy.

    """

    # Makes numpy-based random parts reproducible.
    school_abm.np.random.seed(seed)

    model = school_abm.SchoolModel(seed=seed)

    starter = reset_model_with_starter(model, starter_strategy)

    total_steps = school_abm.N_DAYS * school_abm.TOTAL_DAY_MINUTES

    for _ in range(total_steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe().reset_index(
        names="Collection"
    )

    df["Time (minutes)"] = (
        df["Collection"] * school_abm.DATA_COLLECTION_INTERVAL_MINUTES
    )

    df["Starter strategy"] = starter_strategy
    df["Initial sharer degree"] = model.network.degree(starter.unique_id)
    df["Initial sharer extroversion"] = starter.extroversion

    return df


def run_rq1_experiments(n_runs=N_RUNS, base_seed=BASE_SEED):
    """
    Run all starter strategy experiments.

    """

    strategies = [
        "random",
        "high_degree",
        "low_degree",
        "high_extroversion",
    ]

    all_results = []

    for strategy in strategies:
        print(f"Running RQ1 starter strategy: {strategy}")

        for run_id in tqdm(range(n_runs)):
            seed = base_seed + run_id

            df = run_single_experiment(
                seed=seed,
                starter_strategy=strategy,
            )

            df.insert(0, "Run", run_id + 1)
            df.insert(1, "Seed", seed)

            all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)
    results["Affected"] = results[AFFECTED_STATES].sum(axis=1)

    results.to_csv(OUTPUT_DIR / "rq1_starter_results.csv", index=False)

    return results


def compute_rq1_metrics(results):
    """
    Compute run-level metrics for each starter strategy.

    These metrics correspond to:
    - scale: final affected students
    - speed: time to 50% affected
    - intensity: peak sharers

    """

    metrics = []

    for (strategy, run), group in results.groupby(["Starter strategy", "Run"]):
        group = group.sort_values("Time (minutes)")
        final_row = group.iloc[-1]

        threshold_50 = school_abm.TOTAL_STUDENTS * 0.50
        reached_50 = group[group["Affected"] >= threshold_50]

        if reached_50.empty:
            time_to_50 = None
        else:
            time_to_50 = int(reached_50["Time (minutes)"].iloc[0])

        metrics.append(
            {
                "Starter strategy": strategy,
                "Starter label": STARTER_LABELS[strategy],
                "Run": run,
                "Initial sharer degree": final_row["Initial sharer degree"],
                "Initial sharer extroversion": final_row[
                    "Initial sharer extroversion"
                ],
                "Final affected": final_row["Affected"],
                "Final unaware": final_row[school_abm.STATE_UNAWARE],
                "Final believers": final_row[school_abm.STATE_BELIEVER],
                "Final sharers": final_row[school_abm.STATE_SHARER],
                "Peak sharers": group[school_abm.STATE_SHARER].max(),
                "Time to 50% affected": time_to_50,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "rq1_starter_metrics.csv", index=False)

    summary = metrics_df.groupby(["Starter strategy", "Starter label"]).agg(
        {
            "Initial sharer degree": ["mean", "std"],
            "Initial sharer extroversion": ["mean", "std"],
            "Final affected": ["mean", "std", "min", "max"],
            "Peak sharers": ["mean", "std", "min", "max"],
            "Time to 50% affected": ["mean", "std", "min", "max"],
        }
    )

    summary.to_csv(OUTPUT_DIR / "rq1_starter_summary.csv")

    print()
    print("RQ1 summary metrics:")
    print(summary.round(2))

    return metrics_df, summary


def plot_rq1_starter_comparison(results):
    """
    PLOT average affected students over time by initial spreader type

    """

    plt.figure(figsize=(10, 6))

    for strategy in STARTER_ORDER:
        group = results[results["Starter strategy"] == strategy]

        if group.empty:
            continue

        mean_affected = group.groupby("Time (minutes)")["Affected"].mean()

        plt.plot(
            mean_affected.index,
            mean_affected.values,
            label=STARTER_LABELS[strategy],
        )

    plt.xlabel("Time (minutes)")
    plt.ylabel("Average number of affected students")
    plt.title("Effect of Initial Spreader Type on Misinformation Spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rq1_starter_comparison_affected.png", dpi=200)
    plt.close()


def run_kruskal_wallis_tests(metrics_df):
    """
    Kruskal Wallis tests across starter strategies.

    """

    variables = [
        "Final affected",
        "Peak sharers",
        "Time to 50% affected",
    ]

    test_rows = []

    for variable in variables:
        groups = []

        for strategy in STARTER_ORDER:
            values = metrics_df.loc[
                metrics_df["Starter strategy"] == strategy,
                variable,
            ].dropna()

            if len(values) > 0:
                groups.append(values)

        statistic, p_value = kruskal(*groups)

        test_rows.append(
            {
                "Metric": variable,
                "Kruskal-Wallis H": statistic,
                "p-value": p_value,
            }
        )

    tests_df = pd.DataFrame(test_rows)
    tests_df.to_csv(OUTPUT_DIR / "rq1_kruskal_wallis_tests.csv", index=False)

    with open(OUTPUT_DIR / "rq1_kruskal_wallis_tests.txt", "w", encoding="utf-8") as f:
        f.write("Kruskal-Wallis tests for RQ1 starter experiment\n")
        f.write("=" * 55 + "\n\n")

        for _, row in tests_df.iterrows():
            f.write(f"Metric: {row['Metric']}\n")
            f.write(f"H statistic: {row['Kruskal-Wallis H']:.4f}\n")
            f.write(f"p-value: {row['p-value']:.4f}\n\n")

    print()
    print("Kruskal-Wallis tests:")
    print(tests_df.round(4))

    return tests_df


def main():
    results = run_rq1_experiments(
        n_runs=N_RUNS,
        base_seed=BASE_SEED,
    )

    plot_rq1_starter_comparison(results)

    metrics_df, _summary = compute_rq1_metrics(results)

    run_kruskal_wallis_tests(metrics_df)

    print()
    print("Saved RQ1 outputs in:", OUTPUT_DIR)
    print()
    print("Generated files:")
    print("- rq1_starter_results.csv")
    print("- rq1_starter_metrics.csv")
    print("- rq1_starter_summary.csv")
    print("- rq1_starter_comparison_affected.png")
    print("- rq1_kruskal_wallis_tests.csv")
    print("- rq1_kruskal_wallis_tests.txt")


if __name__ == "__main__":
    main()