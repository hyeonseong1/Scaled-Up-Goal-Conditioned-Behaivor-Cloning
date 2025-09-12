import pandas as pd
import matplotlib.pyplot as plt
import os
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'medium', 'medium, large, giant, teleport')
flags.DEFINE_string('algo', 'gcbc', 'gcbc or gcbcV2')

def main(_):
    base_path = "exp/Offline-RL/Debug/"
    exp_name = f"antmaze-{FLAGS.env}-{FLAGS.algo}"

    seeds = [f"sd{str(i).zfill(3)}" for i in range(23,31)]

    dfs = []
    for seed in seeds:
        file_path = os.path.join(base_path, f"{exp_name}-{seed}", "eval.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["seed"] = seed
            dfs.append(df)
        else:
            print(f"Invalid {file_path}!")

    all_data = pd.concat(dfs, ignore_index=True)

    metrics = [
        "evaluation/task1_success",
        "evaluation/task2_success",
        "evaluation/task3_success",
        "evaluation/task4_success",
        "evaluation/task5_success",
        "evaluation/overall_success"
    ]

    grouped = all_data.groupby("step")[metrics].agg(["mean", "std"])

    summary = pd.DataFrame({
        metric.replace("evaluation/", ""): [
            grouped[(metric, "mean")].mean(),
            grouped[(metric, "mean")].std()
        ]
        for metric in metrics
    }, index=["Mean", "Std"])

    print("\n===== Mean & std for each tasks =====")
    print(summary.round(3))

    # === Save csv ===
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"{exp_name}_summary.csv")
    summary.round(3).to_csv(csv_path)
    print(f"Summary saved to {csv_path}")

    plt.figure(figsize=(10, 6))
    steps = grouped.index

    for metric in metrics:
        mean = grouped[(metric, "mean")]
        std = grouped[(metric, "std")]
        plt.plot(steps, mean, label=metric.replace("evaluation/", ""))
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Success Rate")
    plt.title(f"Task Success Rates (algo: {FLAGS.algo})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{exp_name}')
    # plt.show()

if __name__ == "__main__":
    app.run(main)