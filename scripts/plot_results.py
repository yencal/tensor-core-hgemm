#!/usr/bin/env python3
import sys

import matplotlib.pyplot as plt
import pandas as pd

csv_file = sys.argv[1] if len(sys.argv) > 1 else "hgemm_results.csv"
df = pd.read_csv(csv_file)

plt.figure(figsize=(10, 6))

labels = df["Label"].unique()
for label in sorted(labels):
    data = df[df["Label"] == label]
    plt.plot(
        data["N"], data["TFLOPS"], marker="o", label=label, linewidth=2, markersize=6
    )

sizes = sorted(df["N"].unique())
plt.xticks(sizes, [str(s) for s in sizes])
plt.xlabel("Matrix Size (M=N=K)", fontsize=14)
plt.ylabel("TFLOPS", fontsize=14)
plt.title("HGEMM Performance (FP16)", fontsize=16)
plt.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.grid(True, which="major", alpha=0.3)
plt.tight_layout()

plt.savefig("hgemm_plot.png", dpi=300, bbox_inches="tight")
print("Plot saved: hgemm_plot.png")
