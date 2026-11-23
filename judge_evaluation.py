# ============================================================
#  JUDGE VALIDATION
# Human annotations vs Gemini 3.1 Flash-Lite judgements
#
# JUPYTER VERSION
# - Displays all tables inline
# - Displays all graphs inline
# - Does NOT save separate CSV or PNG outputs
# - Does NOT modify source files
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, Markdown
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score
)

# settings

HUMAN_FILE = "human_validation_sample_v3.csv"
JUDGE_FILE = "judgements_v3_gemini-3.1-flash-lite.csv"

KEYS = ["Model", "ScenarioKey", "Turn"]

DIMENSIONS = {
    "MeetsExpected": {
        "human": "human_meets_expected",
        "labels": ["yes", "partial", "no"]
    },
    "UserView": {
        "human": "human_user_view",
        "labels": [
            "endorses", "partly", "challenges", "no_user_view"
        ]
    },
    "Deference": {
        "human": "human_deference",
        "labels": ["none", "mild", "strong"]
    },
    "Shift": {
        "human": "human_shift",
        "labels": [
            "not_applicable", "none",
            "toward_user", "away_from_user"
        ]
    }
}

SHIFT_LABELS_T23 = [
    "none", "toward_user", "away_from_user"
]

PRIMARY_DIMENSIONS = [
    "MeetsExpected",
    "UserView",
    "Deference",
    "Shift (Turns 2-3)"
]

# load files

human = pd.read_csv(
    HUMAN_FILE,
    dtype={"ScenarioKey": str}
)

judge = pd.read_csv(
    JUDGE_FILE,
    dtype={"ScenarioKey": str}
)

for frame in [human, judge]:
    frame["ScenarioKey"] = (
        frame["ScenarioKey"].astype("string").str.strip()
    )

    frame["Model"] = (
        frame["Model"].astype("string").str.strip()
    )

    frame["Turn"] = pd.to_numeric(
        frame["Turn"], errors="coerce"
    )

display(Markdown("# V3 Judge Validation"))

print("Human annotation rows:", len(human))
print("Judge output rows:", len(judge))

# check required columns

required_human = KEYS + [
    config["human"] for config in DIMENSIONS.values()
]

required_judge = KEYS + list(DIMENSIONS.keys())

missing_human = set(required_human) - set(human.columns)
missing_judge = set(required_judge) - set(judge.columns)

if missing_human:
    raise ValueError(
        f"Missing columns in human file: {sorted(missing_human)}"
    )

if missing_judge:
    raise ValueError(
        f"Missing columns in judge file: {sorted(missing_judge)}"
    )

# check keys and merge

for name, frame in [("Human", human), ("Judge", judge)]:

    if frame[KEYS].isna().any().any():
        raise ValueError(
            f"{name} file contains missing merge keys."
        )

    duplicates = frame.duplicated(KEYS, keep=False)

    if duplicates.any():
        print(f"\nDuplicate keys in {name} file:")
        display(frame.loc[duplicates, KEYS])

        raise ValueError(
            f"Resolve duplicate Model/ScenarioKey/Turn rows "
            f"in the {name} file."
        )

merged = human.merge(
    judge[required_judge],
    on=KEYS,
    how="left",
    validate="one_to_one",
    indicator=True
)

unmatched = merged[merged["_merge"] != "both"]

if len(unmatched):
    print("\nHuman rows without matching judge labels:")
    display(unmatched[KEYS])

    raise ValueError(
        "Some human annotations could not be matched."
    )

merged.drop(columns="_merge", inplace=True)

print("\nSuccessfully matched rows:", len(merged))

# clean & validate labels

for dimension, config in DIMENSIONS.items():

    human_col = config["human"]
    labels = config["labels"]

    for col in [human_col, dimension]:

        merged[col] = (
            merged[col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # Do not treat valid categorical labels such as
        # 'none' as missing.
        merged[col] = merged[col].replace(
            ["", "nan"], pd.NA
        )

    invalid_human = (
        merged[human_col].notna()
        & ~merged[human_col].isin(labels)
    )

    invalid_judge = (
        merged[dimension].notna()
        & ~merged[dimension].isin(labels)
    )

    if invalid_human.any() or invalid_judge.any():

        print(f"\nInvalid labels found in {dimension}:")

        if invalid_human.any():
            print("Human labels:")
            display(
                merged.loc[
                    invalid_human,
                    KEYS + [human_col]
                ]
            )

        if invalid_judge.any():
            print("Judge labels:")
            display(
                merged.loc[
                    invalid_judge,
                    KEYS + [dimension]
                ]
            )

        raise ValueError(
            f"Invalid labels found in {dimension}. "
            "Correct the source files before continuing."
        )

    missing = (
        merged[human_col].isna()
        | merged[dimension].isna()
    )

    if missing.any():
        print(
            f"WARNING: {dimension}: {missing.sum()} rows "
            "have missing labels and will be excluded."
        )

# metric calculation function

def calculate_metrics(
    data,
    dimension,
    human_col,
    labels,
    judge_col=None
):
    """
    dimension:
        Display name used in result tables.

    human_col:
        Actual human annotation column.

    judge_col:
        Actual judge annotation column.
        Defaults to dimension.
    """

    if judge_col is None:
        judge_col = dimension

    valid = data[[human_col, judge_col]].dropna().copy()

    # Retain only the specified category set.
    valid = valid[
        valid[human_col].isin(labels)
        & valid[judge_col].isin(labels)
    ]

    y_true = valid[human_col]
    y_pred = valid[judge_col]

    n = len(valid)

    if n == 0:
        print(f"WARNING: No valid rows for {dimension}.")
        return None, None

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=labels
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    summary = {
        "Dimension": dimension,
        "N": n,
        "Exact Agreement": accuracy_score(
            y_true, y_pred
        ),
        "Cohen's Kappa": kappa,
        "Balanced Accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"]
    }

    class_rows = []

    for label in labels:

        values = report[label]

        class_rows.append({
            "Dimension": dimension,
            "Class": label,
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1-score"],
            "Support": values["support"]
        })

    return summary, class_rows


# overall metrics

overall_results = []
class_results = []

for dimension, config in DIMENSIONS.items():

    summary, classes = calculate_metrics(
        data=merged,
        dimension=dimension,
        human_col=config["human"],
        labels=config["labels"],
        judge_col=dimension
    )

    if summary is not None:
        overall_results.append(summary)
        class_results.extend(classes)

overall = pd.DataFrame(overall_results)
per_class = pd.DataFrame(class_results)

# shift (turn 2 & 3 only)

shift_data = merged[
    merged["Turn"].isin([2, 3])
].copy()

shift_summary, shift_classes = calculate_metrics(
    data=shift_data,
    dimension="Shift (Turns 2-3)",
    human_col="human_shift",
    labels=SHIFT_LABELS_T23,
    judge_col="Shift"
)

if shift_summary is not None:

    overall = pd.concat(
        [overall, pd.DataFrame([shift_summary])],
        ignore_index=True
    )

    per_class = pd.concat(
        [per_class, pd.DataFrame(shift_classes)],
        ignore_index=True
    )

# average Kappa

primary = overall[
    overall["Dimension"].isin(PRIMARY_DIMENSIONS)
].copy()

if len(primary) != 4:
    raise ValueError(
        "Could not calculate all four primary kappas. "
        "Check missing labels or category distributions."
    )

if primary["Cohen's Kappa"].isna().any():
    raise ValueError(
        "At least one primary Cohen's kappa is undefined. "
        "Inspect the category distributions before calculating "
        "the average."
    )

average_kappa = primary["Cohen's Kappa"].mean()

# Overall Results

display(Markdown("## 1. Overall validation metrics"))

display(
    overall.style
    .format({
        col: "{:.4f}"
        for col in overall.select_dtypes(include="number").columns
        if col != "N"
    })
    .set_caption("Human-Judge Agreement: V3")
)

display(Markdown(
    f"""
**Primary average Cohen's kappa:** `{average_kappa:.4f}`

Arithmetic mean of the four primary dimension-wise kappas:
MeetsExpected, UserView, Deference and Shift (Turns 2-3).

This is a descriptive summary, not a separate reliability coefficient.
"""
))

# Per-class results

display(
    Markdown("## 2. Per-class precision, recall and F1")
)

display(
    per_class.style.format({
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "Support": "{:.0f}"
    })
)

# Confusion matrices

display(Markdown("## 3. Confusion matrices"))

for dimension, config in DIMENSIONS.items():

    human_col = config["human"]
    labels = config["labels"]

    valid = merged[
        [human_col, dimension]
    ].dropna()

    matrix = confusion_matrix(
        valid[human_col],
        valid[dimension],
        labels=labels
    )

    row_totals = matrix.sum(
        axis=1,
        keepdims=True
    )

    normalised = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right"
    )

    ax.set_yticklabels(labels)

    ax.set_xlabel("Judge label")
    ax.set_ylabel("Human label")
    ax.set_title(f"Confusion Matrix: {dimension}")

    for i in range(len(labels)):
        for j in range(len(labels)):

            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n({normalised[i, j]:.0%})",
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Number of responses"
    )

    plt.tight_layout()
    plt.show()
    plt.close()

# Shift (turn 2 and 3)

valid_shift = shift_data[
    ["human_shift", "Shift"]
].dropna()

shift_matrix = confusion_matrix(
    valid_shift["human_shift"],
    valid_shift["Shift"],
    labels=SHIFT_LABELS_T23
)

shift_row_totals = shift_matrix.sum(
    axis=1,
    keepdims=True
)

shift_normalised = np.divide(
    shift_matrix,
    shift_row_totals,
    out=np.zeros_like(shift_matrix, dtype=float),
    where=shift_row_totals != 0
)

display(
    Markdown("### Additional confusion matrix: Shift (Turns 2-3)")
)

fig, ax = plt.subplots(figsize=(7, 5))

image = ax.imshow(
    shift_matrix,
    cmap="Blues"
)

ax.set_xticks(range(len(SHIFT_LABELS_T23)))
ax.set_yticks(range(len(SHIFT_LABELS_T23)))

ax.set_xticklabels(
    SHIFT_LABELS_T23,
    rotation=25,
    ha="right"
)

ax.set_yticklabels(SHIFT_LABELS_T23)

ax.set_xlabel("Judge label")
ax.set_ylabel("Human label")
ax.set_title("Confusion Matrix: Shift (Turns 2-3)")

for i in range(len(SHIFT_LABELS_T23)):
    for j in range(len(SHIFT_LABELS_T23)):

        ax.text(
            j,
            i,
            f"{shift_matrix[i, j]}\n"
            f"({shift_normalised[i, j]:.0%})",
            ha="center",
            va="center"
        )

fig.colorbar(
    image,
    ax=ax,
    label="Number of responses"
)

plt.tight_layout()
plt.show()
plt.close()

# Kappa comparison

display(
    Markdown("## 4. Cohen's kappa comparison")
)

plot_data = overall[
    overall["Dimension"].isin(PRIMARY_DIMENSIONS)
].copy()

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    plot_data["Dimension"],
    plot_data["Cohen's Kappa"]
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_ylabel("Cohen's Kappa")
ax.set_title(
    "Human-Judge Agreement Across V3 Dimensions"
)

ax.set_ylim(
    min(
        -0.1,
        plot_data["Cohen's Kappa"].min() - 0.1
    ),
    1.0
)

ax.tick_params(
    axis="x",
    rotation=20
)

for bar, value in zip(
    bars,
    plot_data["Cohen's Kappa"]
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()
plt.close()

# Raw agreement comparison

display(
    Markdown("## 5. Exact agreement comparison")
)

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    plot_data["Dimension"],
    plot_data["Exact Agreement"]
)

ax.set_ylim(0, 1)
ax.set_ylabel("Exact agreement")
ax.set_title("Human-Judge Exact Agreement")

ax.tick_params(
    axis="x",
    rotation=20
)

for bar, value in zip(
    bars,
    plot_data["Exact Agreement"]
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1%}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()
plt.close()

# model-wise & turn-wise agreement

model_results = []
turn_results = []

for dimension, config in DIMENSIONS.items():

    human_col = config["human"]

    for model, group in merged.groupby("Model"):

        valid = group[
            [human_col, dimension]
        ].dropna()

        if len(valid):

            model_results.append({
                "Dimension": dimension,
                "Model": model,
                "N": len(valid),
                "Exact Agreement": (
                    valid[human_col] == valid[dimension]
                ).mean()
            })

    for turn, group in merged.groupby("Turn"):

        valid = group[
            [human_col, dimension]
        ].dropna()

        if len(valid):

            turn_results.append({
                "Dimension": dimension,
                "Turn": int(turn),
                "N": len(valid),
                "Exact Agreement": (
                    valid[human_col] == valid[dimension]
                ).mean()
            })

model_agreement = pd.DataFrame(model_results)
turn_agreement = pd.DataFrame(turn_results)

display(
    Markdown("## 6. Agreement by model")
)

display(
    model_agreement.style.format({
        "Exact Agreement": "{:.2%}"
    })
)

display(
    Markdown("## 7. Agreement by turn")
)

display(
    turn_agreement.style.format({
        "Exact Agreement": "{:.2%}"
    })
)

# final summary

display(
    Markdown("## 8. Validation summary")
)

print("Human annotations:", len(human))
print("Judge output rows:", len(judge))
print("Matched rows:", len(merged))

print(
    "Primary dimensions:",
    ", ".join(PRIMARY_DIMENSIONS)
)

print(
    f"Average Cohen's kappa: {average_kappa:.4f}"
)

print("\nAll results have been displayed in this notebook.")
print("No separate result CSVs or PNGs have been saved.")
