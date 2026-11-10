import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

root = Path("outputs/nanolm/sweeps/char_models")
curves = defaultdict(list)

for log_path in sorted(root.glob("*/eval_log.jsonl")):
    run_name = log_path.parent.name
    model_name = run_name.rsplit("_s", 1)[0]
    seed = run_name.rsplit("_s", 1)[1]

    records = []

    with log_path.open() as file:
        for line in file:
            records.append(json.loads(line))

    curves[model_name].append({
        "seed": seed,
        "records": records,
    })

# Plot individual seed curves
for model_name, runs in curves.items():
    for run in runs:
        tokens = [
            record["train_tokens"]
            for record in run["records"]
        ]

        bpb = [
            record["bpb"]
            for record in run["records"]
        ]

        plt.plot(
            tokens,
            bpb,
            alpha=0.25,
            linewidth=1,
        )

# Plot mean curve across seeds
for model_name, runs in curves.items():
    tokens = [
        record["train_tokens"]
        for record in runs[0]["records"]
    ]

    mean_bpb = []

    for index in range(len(tokens)):
        values = [
            run["records"][index]["bpb"]
            for run in runs
        ]

        mean_bpb.append(sum(values) / len(values))

    plt.plot(
        tokens,
        mean_bpb,
        linewidth=2.5,
        label=model_name,
    )

plt.xlabel("Training tokens seen")
plt.ylabel("Validation bits-per-byte (BPB)")
plt.title("Validation learning curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("validation_learning_curves.png", dpi=200)
plt.show()
