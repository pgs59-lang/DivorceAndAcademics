import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_spss("data.sav", convert_categoricals=False)

# Clean columns
cols = ["P9CURMAR", "X9RSCALK5", "X9MSCALK5"]
for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.loc[df["P9CURMAR"] == -9, "P9CURMAR"] = pd.NA

# Counts after each filter
total_n = len(df)

step1_df = df[df["P9CURMAR"].notna()]
step2_df = step1_df[step1_df["X9RSCALK5"].notna()]
step3_df = step2_df[step2_df["X9MSCALK5"].notna()]

remaining = [
    len(df),
    len(step1_df),
    len(step2_df),
    len(step3_df)
]

removed = [
    0,
    len(df) - len(step1_df),
    len(step1_df) - len(step2_df),
    len(step2_df) - len(step3_df)
]

labels = [
    "Original",
    "After Divorce\nStatus Filter",
    "After Reading\nFilter",
    "After Math\nFilter"
]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, remaining)

plt.title("Observations Remaining After Each Preprocessing Step")
plt.xlabel("Step")
plt.ylabel("Number of Observations")
plt.tight_layout()

for i, bar in enumerate(bars):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{remaining[i]}",
        ha="center",
        va="bottom",
        fontsize=10
    )

    if i > 0:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.85,
            f"Removed:\n{removed[i]}",
            ha="center",
            va="center",
            fontsize=9
        )

plt.savefig("preprocessing_chart.png", dpi=300, bbox_inches="tight")
plt.show()