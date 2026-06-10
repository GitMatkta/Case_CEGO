"""
Chat Room Monitoring — Classification Pipeline
================================================
Simple approach: TF-IDF + Logistic Regression

This script does everything in one file:
  1. Loads the synthetic dataset
  2. Converts text to numbers (TF-IDF)
  3. Adds a few simple handcrafted features
  4. Trains a Logistic Regression to classify category (clean/offensive/gambling/both)
  5. Trains a second Logistic Regression for severity (low/medium/high)
  6. Maps predictions to actions (the business logic from the case brief)
  7. Evaluates everything with plots and metrics
"""

# ─────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────
import json
import warnings
warnings.filterwarnings("ignore")  # Hide sklearn convergence warnings

import numpy as np
import pandas as pd

# Plotting
import matplotlib
matplotlib.use("Agg")  # No GUI, just save to files
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: the ML library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

# ─────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────
PLOTS_DIR = "E:\\Code\\homeProjects\\CEGO\\FTIDF_LogisticRegression\\plots"
DATA_PATH = "E:\\Code\\homeProjects\\CEGO\\FTIDF_LogisticRegression\\chat_messages.json"

import os
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
sns.set_style("whitegrid")


# =====================================================================
# STEP 1: LOAD THE DATA
# =====================================================================
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

# Read the JSON file we generated earlier
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

# Turn it into a pandas DataFrame (basically a spreadsheet in Python)
df = pd.DataFrame(data)

print(f"  Loaded {len(df)} messages")
print(f"  Columns: {list(df.columns)}")
print(f"  Categories: {df['category'].value_counts().to_dict()}")
print()


# =====================================================================
# STEP 2: CONVERT TEXT TO NUMBERS (TF-IDF)
# =====================================================================
print("=" * 60)
print("STEP 2: Converting text to numbers with TF-IDF")
print("=" * 60)

# --- What is TF-IDF? ---
# Computers can't read text — they need numbers.
# TF-IDF (Term Frequency - Inverse Document Frequency) converts each
# message into a row of numbers, where each number represents how
# important a specific word is in that message.
#
# "Term Frequency" = how often a word appears in THIS message
# "Inverse Document Frequency" = words that appear in EVERY message
#   (like "og", "er", "det") get lower scores, because they don't
#   help distinguish between categories.
#
# Example: the word "lort" (shit) appears rarely in clean messages
# but often in offensive ones → it gets a HIGH TF-IDF score in
# offensive messages, which helps the model learn the pattern.

tfidf = TfidfVectorizer(
    max_features=300,    # Only keep the 300 most useful words
    ngram_range=(1, 2),  # Look at single words AND pairs of words
                         # e.g. "hold kæft" as a pair is more
                         # meaningful than "hold" and "kæft" alone
    min_df=2,            # Ignore words that appear only once
    sublinear_tf=True,   # Use log scaling (reduces impact of
                         # very frequent words)
)

# .fit_transform() does two things:
#   1. "fit" = learn which 300 words are most useful
#   2. "transform" = convert every message into a row of 300 numbers
X_tfidf = tfidf.fit_transform(df["content"])

print(f"  Each message is now a row of {X_tfidf.shape[1]} numbers")
print(f"  Example words TF-IDF learned: {tfidf.get_feature_names_out()[:10].tolist()}")
print()


# =====================================================================
# STEP 3: ADD HANDCRAFTED FEATURES
# =====================================================================
print("=" * 60)
print("STEP 3: Adding handcrafted features")
print("=" * 60)

# --- Why handcrafted features? ---
# TF-IDF only looks at which words are used.
# But HOW someone writes also matters:
#   - Angry people use more exclamation marks!!!
#   - Gambling-concerned messages mention money amounts
#   - Offensive messages are often directed at "du" (you) (confirmed in findings, but may be do to dataset bias)
#
# We can capture these patterns with simple counting.

# Feature 1: How long is the message? (number of characters)
df["msg_length"] = df["content"].str.len()

# Feature 2: How many exclamation marks?
df["exclamations"] = df["content"].str.count("!")

# Feature 3: Does the message contain numbers with 3+ digits?
#   (like "5000 kr" or "10.000" — often in gambling messages)
df["has_big_numbers"] = df["content"].str.contains(
    r"\d{3,}", regex=True   # regex pattern: 3 or more digits in a row
).astype(int)               # Convert True/False to 1/0

# Feature 4: Count of known Danish offensive keywords
offensive_keywords = [
    "lort", "fanden", "satan", "helvede", "kæft", "idiot",
    "dum", "pis", "svin", "taber", "smadre", "skrid"
]
df["offensive_word_count"] = df["content"].str.lower().apply(
    lambda msg: sum(1 for word in offensive_keywords if word in msg)
)
# What this does: for each message, go through the keyword list
# and count how many are found. "For fanden, din idiot" → count = 2

# Feature 5: Count of known Danish gambling-concern keywords
gambling_keywords = [
    "tabt", "tabe", "vinde", "spille", "stoppe", "penge",
    "lån", "låne", "huslejen", "desperat", "kontrol"
]
df["gambling_word_count"] = df["content"].str.lower().apply(
    lambda msg: sum(1 for word in gambling_keywords if word in msg)
)

# Combine these 5 features into a matrix
feature_columns = [
    "msg_length", "exclamations", "has_big_numbers",
    "offensive_word_count", "gambling_word_count"
]
X_handcrafted = csr_matrix(df[feature_columns].values.astype(float))
# csr_matrix converts our features into the same sparse format as TF-IDF

# Stack TF-IDF features and handcrafted features side by side
# Each message is now a row of 300 + 5 = 305 numbers
X = hstack([X_tfidf, X_handcrafted])

print(f"  Handcrafted features: {feature_columns}")
print(f"  Total features per message: {X.shape[1]} (300 TF-IDF + 5 handcrafted)")
print()


# =====================================================================
# STEP 4: TRAIN CATEGORY CLASSIFIER
# =====================================================================
print("=" * 60)
print("STEP 4: Training the category classifier")
print("=" * 60)

# --- What is Logistic Regression? ---
# Despite the name, it's used for CLASSIFICATION (not regression).
# It learns a weight for each feature, then combines them:
#   score = w1*feature1 + w2*feature2 + ... + w305*feature305
# If the score is high for "offensive", the message is classified
# as offensive. It's essentially a weighted vote across all features.
#
# Why it's great for this case:
#   - Very fast to train
#   - Easy to explain ("the word 'lort' has weight +2.3 for offensive")
#   - Works well for text classification
#   - Less prone to overfitting than complex models

# First, convert category labels to numbers (sklearn needs numbers)
label_encoder_cat = LabelEncoder()
y_category = label_encoder_cat.fit_transform(df["category"])
category_names = label_encoder_cat.classes_
# Now: "both"=0, "clean"=1, "gambling"=2, "offensive"=3

print(f"  Classes: {list(category_names)}")

# --- Cross-validation ---
# We can't just train on ALL data and test on the SAME data.
# That would be like giving a student the exam answers and then
# testing them on the same questions — of course they'd score 100%.
#
# Instead, we use 5-fold cross-validation:
#   - Split data into 5 equal parts
#   - Train on 4 parts, test on the 1 left out
#   - Repeat 5 times (each part gets to be the test set once)
#   - This gives us an honest performance estimate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# "Stratified" means each fold keeps the same proportion of categories
# (so you don't accidentally get a fold with no "both" examples)

# Train the model
category_model = LogisticRegression(
    max_iter=1000,            # Max training iterations
    C=1.0,                    # Regularization strength (prevents overfitting)
    class_weight="balanced",  # Give more importance to rare classes
                              # (since "both" only has 50 messages vs
                              #  750 for "clean", without this the model
                              #  would just predict "clean" for everything)
    random_state=42,          # Reproducibility
)

# cross_val_predict trains 5 models and returns predictions for EVERY
# message (each predicted by a model that never saw it during training)
y_pred_category = cross_val_predict(category_model, X, y_category, cv=cv)

# Print the results
print("\n  Classification Report:")
print(classification_report(
    y_category, y_pred_category,
    target_names=category_names, digits=3
))


# =====================================================================
# STEP 5: TRAIN SEVERITY CLASSIFIER
# =====================================================================
print("=" * 60)
print("STEP 5: Training the severity classifier")
print("=" * 60)

# Only for messages that are NOT clean (clean messages have no severity)
flagged_mask = df["category"] != "clean"
df_flagged = df[flagged_mask]
X_flagged = X[flagged_mask.values]

# Convert severity labels to numbers
label_encoder_sev = LabelEncoder()
label_encoder_sev.fit(["low", "medium", "high"])
y_severity = label_encoder_sev.transform(df_flagged["severity"])
severity_names = label_encoder_sev.classes_

print(f"  Flagged messages: {len(df_flagged)}")
print(f"  Severity classes: {list(severity_names)}")

# Same approach: Logistic Regression with cross-validation
severity_model = LogisticRegression(
    max_iter=1000, C=1.0, class_weight="balanced", random_state=42
)
cv_sev = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_severity = cross_val_predict(
    severity_model, X_flagged, y_severity, cv=cv_sev
)

print("\n  Severity Classification Report:")
print(classification_report(
    y_severity, y_pred_severity,
    target_names=severity_names, digits=3
))


# =====================================================================
# STEP 6: ACTION ENGINE (Business Logic)
# =====================================================================
print("=" * 60)
print("STEP 6: Action engine")
print("=" * 60)

# --- This is the simple part: a lookup table ---
# Given a category + severity, what should the system DO?
# This comes directly from the case brief's table.

ACTION_TABLE = {
    ("clean", "none"):       "Ingen handling",
    ("offensive", "low"):    "Send advarsel i chatforum",
    ("offensive", "medium"): "Opret sag i backoffice",
    ("offensive", "high"):   "Blokér brugeren fra chat",
    ("gambling", "low"):     "Send mail om ansvarligt spil",
    ("gambling", "medium"):  "Opret sag i backoffice",
    ("gambling", "high"):    "Sæt spilgrænser / selvudelukkelse",
    ("both", "low"):         "Send advarsel + mail om ansvarligt spil",
    ("both", "medium"):      "Opret sag i backoffice (begge)",
    ("both", "high"):        "Blokér + Sæt spilgrænser",
}

def get_action(category, severity):
    """Look up the action for a given category + severity."""
    return ACTION_TABLE.get((category, severity), "Opret sag i backoffice")

# Apply predictions to the dataframe
pred_cat_labels = label_encoder_cat.inverse_transform(y_pred_category)
df["pred_category"] = pred_cat_labels

# For severity: only flagged messages get a severity prediction
df["pred_severity"] = "none"
flagged_indices = df[flagged_mask].index
pred_sev_labels = label_encoder_sev.inverse_transform(y_pred_severity)
df.loc[flagged_indices, "pred_severity"] = pred_sev_labels

# Determine actions
df["pred_action"] = df.apply(
    lambda row: get_action(row["pred_category"], row["pred_severity"]),
    axis=1
)
df["true_action"] = df.apply(
    lambda row: get_action(row["category"], row["severity"]),
    axis=1
)

action_accuracy = (df["pred_action"] == df["true_action"]).mean()
print(f"  Action accuracy: {action_accuracy:.1%}")
print(f"  (This means {action_accuracy:.1%} of the time, the system")
print(f"   would take the CORRECT action)")
print()


# =====================================================================
# STEP 7: EVALUATION & PLOTS
# =====================================================================
print("=" * 60)
print("STEP 7: Generating evaluation plots")
print("=" * 60)
print(list(label_encoder_sev.classes_))
# ── Plot 1: Confusion Matrix for Categories ──────────────────────────
# A confusion matrix shows WHERE the model makes mistakes.
# Rows = what the message actually is
# Columns = what the model predicted
# Diagonal = correct predictions (we want these to be high)
# Off-diagonal = mistakes

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

cm_cat = confusion_matrix(y_category, y_pred_category)
sns.heatmap(cm_cat, annot=True, fmt="d", cmap="Blues",
            xticklabels=category_names, yticklabels=category_names,
            ax=axes[0])
axes[0].set_title("Category Confusion Matrix", fontweight="bold", fontsize=13)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

cm_sev = confusion_matrix(y_severity, y_pred_severity)
logical_order = ["low", "medium", "high"]

cm_sev = confusion_matrix(y_severity, y_pred_severity,
                           labels=[0, 1, 2])

sns.heatmap(cm_sev, annot=True, fmt="d", cmap="Oranges",
            xticklabels=logical_order, yticklabels=logical_order,
            ax=axes[1])
axes[1].set_title("Severity Confusion Matrix", fontweight="bold", fontsize=13)
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrices.png", bbox_inches="tight")
plt.close()
print("  Saved: confusion_matrices.png")

# ── Plot 2: Feature Importances ──────────────────────────────────────
# Logistic Regression gives us weights (coefficients) for each feature.
# High positive weight = strong signal for that class.
# This is one of the BIG advantages over complex models like XGBoost.

# Train on full data to get the final weights
category_model.fit(X, y_category)

# Get feature names
tfidf_feature_names = tfidf.get_feature_names_out().tolist()
all_feature_names = tfidf_feature_names + feature_columns

# For each class, find the top 10 most important features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (cat_name, ax) in enumerate(zip(category_names, axes.flat)):
    # category_model.coef_[idx] contains the weight for each feature
    # for this specific class
    weights = category_model.coef_[idx]
    top_indices = np.argsort(np.abs(weights))[-10:]  # top 10 by magnitude
    top_names = [all_feature_names[i] for i in top_indices]
    top_weights = weights[top_indices]

    colors = ["#e74c3c" if w < 0 else "#2ecc71" for w in top_weights]
    ax.barh(top_names, top_weights, color=colors)
    ax.set_title(f'Top features for "{cat_name}"', fontweight="bold")
    ax.set_xlabel("Weight (green = supports this class, red = opposes)")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_importances.png", bbox_inches="tight")
plt.close()
print("  Saved: feature_importances.png")

# ── Plot 3: Dataset distributions ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors_cat = {"clean": "#2ecc71", "offensive": "#e74c3c",
              "gambling": "#f39c12", "both": "#8e44ad"}
cat_counts = df["category"].value_counts()
axes[0].bar(cat_counts.index, cat_counts.values,
            color=[colors_cat[c] for c in cat_counts.index])
axes[0].set_title("Messages by Category", fontweight="bold")
axes[0].set_ylabel("Count")
for i, (cat, val) in enumerate(cat_counts.items()):
    axes[0].text(i, val + 5, str(val), ha="center", fontsize=9)

colors_sev = {"none": "#bdc3c7", "low": "#f1c40f",
              "medium": "#e67e22", "high": "#c0392b"}
sev_order = ["none", "low", "medium", "high"]
sev_counts = [len(df[df["severity"] == s]) for s in sev_order]
axes[1].bar(sev_order, sev_counts,
            color=[colors_sev[s] for s in sev_order])
axes[1].set_title("Messages by Severity", fontweight="bold")
axes[1].set_ylabel("Count")
for i, val in enumerate(sev_counts):
    axes[1].text(i, val + 5, str(val), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/distributions.png", bbox_inches="tight")
plt.close()
print("  Saved: distributions.png")


# =====================================================================
# STEP 8: ERROR ANALYSIS
# =====================================================================
print("\n" + "=" * 60)
print("STEP 8: Error analysis")
print("=" * 60)

total_errors = (y_pred_category != y_category).sum()
print(f"\n  Total misclassifications: {total_errors} / {len(df)} "
      f"({total_errors/len(df)*100:.1f}%)")

# The MOST important errors: high-severity messages missed as clean
critical_fn = df[
    (df["category"].isin(["gambling", "both"])) &
    (df["severity"] == "high") &
    (df["pred_category"] == "clean")
]
print(f"  CRITICAL false negatives (high gambling → clean): {len(critical_fn)}")

# Less important but annoying: clean messages falsely flagged
false_positives = df[
    (df["category"] == "clean") & (df["pred_category"] != "clean")
]
print(f"  False positives (clean → flagged): {len(false_positives)}")

# Show some example mistakes
errors = df[df["pred_category"] != df["category"]]
if len(errors) > 0:
    print(f"\n  Example mistakes:")
    sample = errors.head(10)
    for _, row in sample.iterrows():
        print(f"    [{row['category']:10s} → {row['pred_category']:10s}] "
              f"{row['content'][:65]}")

# ── Save results ─────────────────────────────────────────────────────
results = classification_report(
    y_category, y_pred_category,
    target_names=category_names, output_dict=True
)
results["action_accuracy"] = action_accuracy
results["total_errors"] = int(total_errors)
results["critical_false_negatives"] = int(len(critical_fn))
results["false_positives_clean"] = int(len(false_positives))

with open(f"{PLOTS_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

df.to_csv(f"{PLOTS_DIR}/predictions.csv", index=False, encoding="utf-8")

print(f"\n  Results saved to {PLOTS_DIR}/")
print("\nDone! ✓")
