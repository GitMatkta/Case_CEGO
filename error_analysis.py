"""
Error Analysis — Full Breakdown
=================================
Loads the predictions CSV and prints a human-readable breakdown of:
  1. All 65 misclassifications grouped by error type
  2. The 40 false positives (clean → flagged) in detail
"""

import pandas as pd

# ── Load predictions ───────────────────────────────────────────────
df = pd.read_csv("E:\Code\homeProjects\CEGO\FTIDF_LogisticRegression\predictions.csv")

# Separate correct vs wrong
errors = df[df["pred_category"] != df["category"]].copy()

# ══════════════════════════════════════════════════════════════════
# PART 1 — ALL 65 ERRORS
# grouped by: what it actually was → what the model said it was
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"PART 1 — ALL {len(errors)} MISCLASSIFICATIONS")
print("=" * 70)

# Group errors by the (true → predicted) pair
error_types = errors.groupby(["category", "pred_category"])

for (true_cat, pred_cat), group in error_types:
    print(f"\n{'─' * 70}")
    print(f"  ACTUAL: {true_cat.upper()}  →  PREDICTED: {pred_cat.upper()}")
    print(f"  Count: {len(group)}")
    print(f"{'─' * 70}")
    for _, row in group.iterrows():
        sev_label = f"[sev={row['severity']}]" if row["severity"] != "none" else ""
        print(f"  msg_{row['msg_id']:04d} | user_{row['user_id']:2d} {sev_label}")
        print(f"  → \"{row['content']}\"")
        print()

# Summary count table
print("\n" + "=" * 70)
print("ERROR TYPE SUMMARY")
print("=" * 70)
summary = errors.groupby(["category", "pred_category"]).size().reset_index(name="count")
summary.columns = ["Actual", "Predicted as", "Count"]
summary = summary.sort_values("Count", ascending=False)
print(summary.to_string(index=False))


# ══════════════════════════════════════════════════════════════════
# PART 2 — FALSE POSITIVES (clean → something else)
# ══════════════════════════════════════════════════════════════════
false_positives = errors[errors["category"] == "clean"].copy()

print("\n\n" + "=" * 70)
print(f"PART 2 — FALSE POSITIVES: {len(false_positives)} CLEAN MESSAGES WRONGLY FLAGGED")
print("Clean messages the model mistakenly thought were problematic.")
print("=" * 70)

# Group by what the model confused them for
for pred_cat, group in false_positives.groupby("pred_category"):
    print(f"\n{'─' * 70}")
    print(f"  Clean messages flagged as: {pred_cat.upper()} (n={len(group)})")
    print(f"{'─' * 70}")
    for _, row in group.iterrows():
        print(f"  msg_{row['msg_id']:04d} | user_{row['user_id']:2d}")
        print(f"  → \"{row['content']}\"")
        print()

# Quick pattern summary
print("=" * 70)
print("WHY DID THESE GET FLAGGED? — Pattern Notes")
print("=" * 70)
print("""
  Clean → gambling (most common):
    The model sees words like 'spille', 'stoppe', 'ansvarligt' and thinks
    gambling. But these are actually RESPONSIBLE messages ("spil ansvarligt",
    "nu stopper jeg"). The model doesn't understand context — it sees the
    words, not the meaning.

  Clean → offensive:
    Messages using Danish exclamations like 'hold kæft' in a POSITIVE way
    ("hold kæft det var godt!") get flagged because the phrase normally
    signals offensive content. Classic sarcasm/slang problem.

  Clean → both:
    Messages combining gambling vocabulary with exclamations, triggering
    both signals at once. E.g. "Nå men så slår jeg mig selv ihjel med
    det her spil" — metaphor, not literal, not problematic.
""")
