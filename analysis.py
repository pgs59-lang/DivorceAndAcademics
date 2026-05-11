import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load data
# ----------------------------
# convert_categoricals=False keeps the raw SPSS codes
df = pd.read_spss("data.sav", convert_categoricals=False)

# ----------------------------
# 2. Clean important columns
# ----------------------------
score_cols = [
    "X1RSCALK5", "X1MSCALK5",
    "X4RSCALK5", "X4MSCALK5",
    "X9RSCALK5", "X9MSCALK5"
]
marital_cols = ["P1CURMAR", "P4CURMAR", "P9CURMAR"]

for col in score_cols + marital_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------
# 3. Create divorce flags
# ----------------------------
# In your file:
# 3 = divorced
# -9 = not ascertained
# NaN = missing

for col in marital_cols:
    df.loc[df[col] == -9, col] = pd.NA

df["divorced_p1"] = (df["P1CURMAR"] == 3).astype("float")
df["divorced_p4"] = (df["P4CURMAR"] == 3).astype("float")
df["divorced_p9"] = (df["P9CURMAR"] == 3).astype("float")

# Set divorce flags to missing when marital status is missing
df.loc[df["P1CURMAR"].isna(), "divorced_p1"] = pd.NA
df.loc[df["P4CURMAR"].isna(), "divorced_p4"] = pd.NA
df.loc[df["P9CURMAR"].isna(), "divorced_p9"] = pd.NA

# ----------------------------
# 4. Create average score variables
# ----------------------------
df["GRADE4_AVG"] = df[["X4RSCALK5", "X4MSCALK5"]].mean(axis=1)
df["GRADE9_AVG"] = df[["X9RSCALK5", "X9MSCALK5"]].mean(axis=1)

# ----------------------------
# 5. Basic checks
# ----------------------------
print("\n=== Divorce group counts ===")
print("P1:")
print(df["divorced_p1"].value_counts(dropna=False))
print("\nP4:")
print(df["divorced_p4"].value_counts(dropna=False))
print("\nP9:")
print(df["divorced_p9"].value_counts(dropna=False))

print("\n=== Missing values in score columns ===")
print(df[score_cols].isna().sum())

# ----------------------------
# 6. Summary table for Wave 9
# ----------------------------
wave9 = df[df["divorced_p9"].isin([0, 1])].copy()
wave9["divorce_group"] = wave9["divorced_p9"].map({0: "Not Divorced", 1: "Divorced"})

summary_wave9 = wave9.groupby("divorce_group")[["X9RSCALK5", "X9MSCALK5", "GRADE9_AVG"]].agg(["mean", "count"])
print("\n=== Wave 9 summary by divorce group ===")
print(summary_wave9)

# ----------------------------
# 7. Graph 1: Bar chart of Wave 9 averages
# ----------------------------
# Keep only rows with valid wave 9 reading and math scores
wave9_plot = wave9.dropna(subset=["X9RSCALK5", "X9MSCALK5"]).copy()

# Average scores by group
bar_data = wave9_plot.groupby("divorce_group")[["X9RSCALK5", "X9MSCALK5"]].mean()
bar_data = bar_data.rename(columns={
    "X9RSCALK5": "Reading",
    "X9MSCALK5": "Math"
})

# Number of individuals in each group
group_counts = wave9_plot.groupby("divorce_group").size()

# Build x-axis labels with n counts
label_order = list(bar_data.index)
x_labels = [f"{group}\n(n={group_counts[group]})" for group in label_order]

# Plot
ax = bar_data.plot(kind="bar", figsize=(8, 5))
ax.set_title("Wave 9 Average Scores by Divorce Group")
ax.set_xlabel("Divorce Group")
ax.set_ylabel("Average Score")
ax.set_xticklabels(x_labels, rotation=0)

# Add count labels above each group
group_max = bar_data.max(axis=1)
for i, group in enumerate(label_order):
    ax.text(
        i,
        group_max[group] + 1.5,
        f"n={group_counts[group]}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.savefig("wave9_average_scores_by_divorce_group_with_counts.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------
# 8. Graph 2: Boxplot of Wave 9 reading scores
# ----------------------------
box_reading = [
    wave9.loc[wave9["divorce_group"] == "Not Divorced", "X9RSCALK5"].dropna(),
    wave9.loc[wave9["divorce_group"] == "Divorced", "X9RSCALK5"].dropna()
]

plt.figure(figsize=(8, 5))
plt.boxplot(box_reading, tick_labels=["Not Divorced", "Divorced"])
plt.title("Wave 9 Reading Score Distribution by Divorce Group")
plt.xlabel("Divorce Group")
plt.ylabel("Reading Score")
plt.tight_layout()
plt.savefig("wave9_reading_boxplot_by_divorce_group.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------
# 9. Graph 3: Boxplot of Wave 9 math scores
# ----------------------------
box_math = [
    wave9.loc[wave9["divorce_group"] == "Not Divorced", "X9MSCALK5"].dropna(),
    wave9.loc[wave9["divorce_group"] == "Divorced", "X9MSCALK5"].dropna()
]

plt.figure(figsize=(8, 5))
plt.boxplot(box_math, tick_labels=["Not Divorced", "Divorced"])
plt.title("Wave 9 Math Score Distribution by Divorce Group")
plt.xlabel("Divorce Group")
plt.ylabel("Math Score")
plt.tight_layout()
plt.savefig("wave9_math_boxplot_by_divorce_group.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------
# 10. Graph 4: Line graph from Wave 4 to Wave 9
# ----------------------------
line_data = df[df["divorced_p9"].isin([0, 1])].copy()
line_data["divorce_group"] = line_data["divorced_p9"].map({0: "Not Divorced", 1: "Divorced"})

reading_line = line_data.groupby("divorce_group")[["X4RSCALK5", "X9RSCALK5"]].mean()
math_line = line_data.groupby("divorce_group")[["X4MSCALK5", "X9MSCALK5"]].mean()

# Reading line graph
plt.figure(figsize=(8, 5))
for group in reading_line.index:
    plt.plot(["Wave 4", "Wave 9"], reading_line.loc[group], marker="o", label=group)

plt.title("Reading Scores Over Time by Divorce Group")
plt.xlabel("Wave")
plt.ylabel("Average Reading Score")
plt.legend(title="Divorce Group")
plt.tight_layout()
plt.savefig("reading_scores_over_time_by_divorce_group.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Math line graph
plt.figure(figsize=(8, 5))
for group in math_line.index:
    plt.plot(["Wave 4", "Wave 9"], math_line.loc[group], marker="o", label=group)

plt.title("Math Scores Over Time by Divorce Group")
plt.xlabel("Wave")
plt.ylabel("Average Math Score")
plt.legend(title="Divorce Group")
plt.tight_layout()
plt.savefig("math_scores_over_time_by_divorce_group.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------
# 11. Optional: timing of divorce
# ----------------------------
def timing_group(row):
    p1 = row["P1CURMAR"]
    p4 = row["P4CURMAR"]
    p9 = row["P9CURMAR"]

    if pd.isna(p1) or pd.isna(p4) or pd.isna(p9):
        return pd.NA

    if p1 != 3 and p4 != 3 and p9 != 3:
        return "Never Divorced"
    elif p1 == 3 or p4 == 3:
        return "Earlier Divorce"
    elif p1 != 3 and p4 != 3 and p9 == 3:
        return "Later Divorce"
    else:
        return pd.NA

df["divorce_timing"] = df.apply(timing_group, axis=1)

timing_data = df[df["divorce_timing"].notna()].copy()
timing_summary = timing_data.groupby("divorce_timing")[["X9RSCALK5", "X9MSCALK5"]].mean()
print("\n=== Wave 9 averages by timing of divorce ===")
print(timing_summary)

timing_summary = timing_summary.rename(columns={
    "X9RSCALK5": "Reading",
    "X9MSCALK5": "Math"
})

plt.figure(figsize=(8, 5))
timing_summary.plot(kind="bar", figsize=(8, 5))
plt.title("Wave 9 Scores by Timing of Divorce")
plt.xlabel("Timing of Divorce")
plt.ylabel("Average Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("wave9_scores_by_timing_of_divorce.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


def plot_wave_average_scores(df, divorce_col, reading_col, writing_col, wave_name):
    plot_df = df[df[divorce_col].isin([0, 1])].copy()
    plot_df["divorce_group"] = plot_df[divorce_col].map({
        0: "Not Divorced",
        1: "Divorced"
    })

    avg_scores = plot_df.groupby("divorce_group")[[reading_col, writing_col]].mean()
    avg_scores = avg_scores.rename(columns={
        reading_col: "Reading",
        writing_col: "Writing"   # change to "Math" if you want the true label
    })

    ax = avg_scores.plot(kind="bar", figsize=(8, 5))
    ax.set_title(f"{wave_name} Average Scores by Divorce Group")
    ax.set_xlabel("Divorce Group")
    ax.set_ylabel("Average Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()

plot_wave_average_scores(df, "divorced_p1", "X1RSCALK5", "X1MSCALK5", "Wave 1")
plot_wave_average_scores(df, "divorced_p4", "X4RSCALK5", "X4MSCALK5", "Wave 4")
plot_wave_average_scores(df, "divorced_p9", "X9RSCALK5", "X9MSCALK5", "Wave 9")