from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal
from tqdm import tqdm

import school_abm


OUTPUT_DIR = Path("RQ02_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

N_RUNS = 50 #change here =10 for testing
BASE_SEED = 42

AFFECTED_STATES = [
    school_abm.STATE_EXPOSED,
    school_abm.STATE_BELIEVER,
    school_abm.STATE_SHARER,
]

SCENARIO_LABELS = {
    "baseline_schedule": "Baseline schedule",
    "uniform_schedule": "Uniform interaction schedule",
    "high_break_lunch": "High break/lunch interaction",
    "low_break_lunch": "Low break/lunch interaction",
}

SCENARIO_ORDER = [
    "low_break_lunch",
    "uniform_schedule",
    "baseline_schedule",
    "high_break_lunch",
]


def copy_school_day_with_multipliers(multipliers):
    """
    Create a modified copy of the original school-day schedule.
    """

    new_schedule = []

    for period in school_abm.SCHOOL_DAY:
        new_period = period.copy()
        period_name = new_period["name"]

        if period_name in multipliers:
            new_period["interaction_multiplier"] = multipliers[period_name]

        new_schedule.append(new_period)

    return new_schedule


def average_baseline_multiplier():

    total_duration = sum(period["duration_min"] for period in school_abm.SCHOOL_DAY)

    weighted_sum = sum(
        period["duration_min"] * period["interaction_multiplier"]
        for period in school_abm.SCHOOL_DAY
    )

    return weighted_sum / total_duration


def create_scenarios():
    """
    schedule scenarios for RQ2

    """

    baseline_schedule = [period.copy() for period in school_abm.SCHOOL_DAY]

    uniform_multiplier = average_baseline_multiplier()

    uniform_schedule = [
        {
            **period,
            "interaction_multiplier": uniform_multiplier,
        }
        for period in school_abm.SCHOOL_DAY
    ]

    high_break_lunch_schedule = copy_school_day_with_multipliers(
        {
            "lesson": 0.2,
            "short_break": 1.5,
            "lunch": 2.0,
        }
    )

    low_break_lunch_schedule = copy_school_day_with_multipliers(
        {
            "lesson": 0.2,
            "short_break": 0.4,
            "lunch": 0.6,
        }
    )

    return {
        "baseline_schedule": baseline_schedule,
        "uniform_schedule": uniform_schedule,
        "high_break_lunch": high_break_lunch_schedule,
        "low_break_lunch": low_break_lunch_schedule,
    }


def run_single_scenario(scenario_name, schedule, n_runs=N_RUNS, base_seed=BASE_SEED):
    """
    replace the school schedule, run simulations,
    and restore the original schedule afterwards.

    """

    original_schedule = school_abm.SCHOOL_DAY
    original_total_day_minutes = school_abm.TOTAL_DAY_MINUTES

    try:
        school_abm.SCHOOL_DAY = schedule
        school_abm.TOTAL_DAY_MINUTES = sum(
            period["duration_min"] for period in school_abm.SCHOOL_DAY
        )

        all_runs = []

        for run_id in tqdm(range(n_runs), desc=scenario_name):
            seed = base_seed + run_id

            # Reproducibility for numpy-based random parts.
            school_abm.np.random.seed(seed)

            run_df = school_abm.run_simulation(seed=seed)

            run_df.insert(0, "Run", run_id + 1)
            run_df.insert(1, "Seed", seed)
            run_df.insert(2, "Scenario", scenario_name)
            run_df.insert(3, "Scenario label", SCENARIO_LABELS[scenario_name])

            all_runs.append(run_df)

        results = pd.concat(all_runs, ignore_index=True)
        results["Affected"] = results[AFFECTED_STATES].sum(axis=1)

        results.to_csv(
            OUTPUT_DIR / f"{scenario_name}_results.csv",
            index=False,
        )

        return results

    finally:
        school_abm.SCHOOL_DAY = original_schedule
        school_abm.TOTAL_DAY_MINUTES = original_total_day_minutes


def run_rq2_experiments():

    scenarios = create_scenarios()
    all_results = []

    for scenario_name, schedule in scenarios.items():
        print(f"Running RQ2 scenario: {scenario_name}")

        results = run_single_scenario(
            scenario_name=scenario_name,
            schedule=schedule,
            n_runs=N_RUNS,
            base_seed=BASE_SEED,
        )

        all_results.append(results)

    combined_results = pd.concat(all_results, ignore_index=True)

    combined_results.to_csv(
        OUTPUT_DIR / "rq2_schedule_results.csv",
        index=False,
    )

    return combined_results


def compute_rq2_metrics(results):
    """
    metrics for each schedule scenario

    """

    metrics = []

    for (scenario, run), group in results.groupby(["Scenario", "Run"]):
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
                "Scenario": scenario,
                "Scenario label": SCENARIO_LABELS[scenario],
                "Run": run,
                "Final affected": final_row["Affected"],
                "Final unaware": final_row[school_abm.STATE_UNAWARE],
                "Final believers": final_row[school_abm.STATE_BELIEVER],
                "Final sharers": final_row[school_abm.STATE_SHARER],
                "Peak sharers": group[school_abm.STATE_SHARER].max(),
                "Time to 50% affected": time_to_50,
            }
        )

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        OUTPUT_DIR / "rq2_schedule_metrics.csv",
        index=False,
    )

    summary = metrics_df.groupby(["Scenario", "Scenario label"]).agg(
        {
            "Final affected": ["mean", "std", "min", "max"],
            "Peak sharers": ["mean", "std", "min", "max"],
            "Time to 50% affected": ["mean", "std", "min", "max"],
        }
    )

    summary.to_csv(OUTPUT_DIR / "rq2_schedule_summary.csv")

    print()
    print("RQ2 summary metrics:")
    print(summary.round(2))

    return metrics_df, summary


def plot_rq2_schedule_comparison(results):
    """
    RQ2 plot:
    average affected students over time by schedule scenario.

    """

    plt.figure(figsize=(10, 6))

    for scenario in SCENARIO_ORDER:
        group = results[results["Scenario"] == scenario]

        if group.empty:
            continue

        mean_affected = group.groupby("Time (minutes)")["Affected"].mean()

        plt.plot(
            mean_affected.index,
            mean_affected.values,
            label=SCENARIO_LABELS[scenario],
        )

    plt.xlabel("Time (minutes)")
    plt.ylabel("Average number of affected students")
    plt.title("Effect of School-Day Interaction Patterns on Misinformation Spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "rq2_schedule_comparison_affected.png",
        dpi=200,
    )
    plt.close()


def run_kruskal_wallis_tests(metrics_df):
    """
    Kruskal-Wallis 
    
    """

    variables = {
        "Final affected": "Final misinformation reach",
        "Peak sharers": "Peak number of sharers",
        "Time to 50% affected": "Time to 50% reach",
    }

    test_rows = []

    for variable, label in variables.items():
        groups = []

        for scenario in SCENARIO_ORDER:
            values = metrics_df.loc[
                metrics_df["Scenario"] == scenario,
                variable,
            ].dropna()

            if len(values) > 0:
                groups.append(values)

        statistic, p_value = kruskal(*groups)

        test_rows.append(
            {
                "Metric": label,
                "Kruskal-Wallis H": statistic,
                "p-value": p_value,
            }
        )

    tests_df = pd.DataFrame(test_rows)

    tests_df.to_csv(
        OUTPUT_DIR / "rq2_kruskal_wallis_tests.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "rq2_kruskal_wallis_tests.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("Kruskal-Wallis tests for RQ2 schedule experiment\n")
        file.write("=" * 55 + "\n\n")

        for _, row in tests_df.iterrows():
            file.write(f"Metric: {row['Metric']}\n")
            file.write(f"H statistic: {row['Kruskal-Wallis H']:.4f}\n")
            file.write(f"p-value: {row['p-value']:.4g}\n\n")

    print()
    print("Kruskal-Wallis tests:")
    print(tests_df.round(4))

    return tests_df


def main():
    results = run_rq2_experiments()

    plot_rq2_schedule_comparison(results)

    metrics_df, _summary = compute_rq2_metrics(results)

    run_kruskal_wallis_tests(metrics_df)

    print()
    print("Saved RQ2 outputs in:", OUTPUT_DIR)
    print()
    print("Generated files:")
    print("- rq2_schedule_results.csv")
    print("- rq2_schedule_metrics.csv")
    print("- rq2_schedule_summary.csv")
    print("- rq2_schedule_comparison_affected.png")
    print("- rq2_kruskal_wallis_tests.csv")
    print("- rq2_kruskal_wallis_tests.txt")


if __name__ == "__main__":
    main()