"""
Chat Room Monitoring — Deep EDA
================================
A thorough exploration of the synthetic dataset across 8 angles:

  1. Dataset overview & class balance
  2. Message-level text statistics
  3. Vocabulary & top words per category
  4. User-level behavior profiles
  5. Temporal patterns
  6. Escalation & severity progression
  7. Edge case & ambiguity analysis
  8. Actionability: what actions would actually be triggered
"""

import json
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# ─────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────
DATA_PATH = "E:\\Code\\homeProjects\\CEGO\\FTIDF_LogisticRegression\\chat_messages.json"
OUT       = "E:\\Code\\homeProjects\\CEGO\\FTIDF_LogisticRegression\\eda_output"

import os
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"figure.dpi": 140, "font.size": 10})
sns.set_style("whitegrid")

COLORS = {
    "clean":     "#2ecc71",
    "offensive": "#e74c3c",
    "gambling":  "#f39c12",
    "both":      "#8e44ad",
    "none":      "#bdc3c7",
    "low":       "#f1c40f",
    "medium":    "#e67e22",
    "high":      "#c0392b",
}
CAT_ORDER = ["clean", "offensive", "gambling", "both"]
SEV_ORDER = ["none", "low", "medium", "high"]

# ─────────────────────────────────────────────────────────────────────
# LOAD & ENRICH
# ─────────────────────────────────────────────────────────────────────
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Text-level features
df["char_count"]    = df["content"].str.len()
df["word_count"]    = df["content"].str.split().str.len()
df["exclamations"]  = df["content"].str.count("!")
df["questions"]     = df["content"].str.count(r"\?")
df["caps_ratio"]    = df["content"].apply(
    lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
)
df["emoji_count"]   = df["content"].str.count(
    r"[\U0001F300-\U0001FFFF\u2600-\u27BF]"
)
df["has_number"]    = df["content"].str.contains(r"\b\d{3,}\b", regex=True)
df["has_kr"]        = df["content"].str.lower().str.contains(r"\d+\s*kr", regex=True)
df["is_flagged"]    = df["category"] != "clean"
df["day"]           = df["timestamp"].dt.date
df["hour"]          = df["timestamp"].dt.hour
df["weekday"]       = df["timestamp"].dt.day_name()
df["msg_rank"]      = df.groupby("user_id").cumcount()          # position in user's history
df["user_msg_total"]= df.groupby("user_id")["msg_id"].transform("count")

# Keyword counts
OFFENSIVE_KW = ["lort","fanden","satan","helvede","kæft","idiot","dum","pis",
                "svin","taber","smadre","skrid","slå","tæsk","luder","kujon"]
GAMBLING_KW  = ["tabt","tabe","vinde","stoppe","penge","lån","huslejen",
                "desperat","kontrol","forbrugslån","kan ikke","skjuler",
                "lyver","familie","regning","dobler"]

df["n_off_kw"] = df["content"].str.lower().apply(
    lambda t: sum(1 for w in OFFENSIVE_KW if w in t)
)
df["n_gam_kw"] = df["content"].str.lower().apply(
    lambda t: sum(1 for w in GAMBLING_KW if w in t)
)

print(f"Dataset loaded: {len(df)} messages, {df['user_id'].nunique()} users")


# ══════════════════════════════════════════════════════════════════════
# 1. DATASET OVERVIEW & CLASS BALANCE
# ══════════════════════════════════════════════════════════════════════
print("\n[1/8] Dataset overview & class balance...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("1 · Dataset Overview & Class Balance", fontsize=14, fontweight="bold", y=1.01)

# 1a: Category counts + %
cat_counts = df["category"].value_counts().reindex(CAT_ORDER)
bars = axes[0].bar(CAT_ORDER, cat_counts, color=[COLORS[c] for c in CAT_ORDER], edgecolor="white")
axes[0].set_title("Category Distribution")
axes[0].set_ylabel("Message count")
for bar, (cat, n) in zip(bars, cat_counts.items()):
    pct = n / len(df) * 100
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 6, f"{n}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=9)

# 1b: Severity distribution
sev_counts = df["severity"].value_counts().reindex(SEV_ORDER)
bars2 = axes[1].bar(SEV_ORDER, sev_counts, color=[COLORS[s] for s in SEV_ORDER], edgecolor="white")
axes[1].set_title("Severity Distribution")
axes[1].set_ylabel("Message count")
for bar, n in zip(bars2, sev_counts):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 6, str(n), ha="center", fontsize=9)

# 1c: Category × Severity stacked bar
pivot = pd.crosstab(df["category"], df["severity"]).reindex(
    CAT_ORDER, axis=0).reindex(SEV_ORDER, axis=1).fillna(0)
pivot.drop(columns="none", errors="ignore").plot(
    kind="bar", stacked=True, ax=axes[2],
    color=[COLORS[s] for s in ["low","medium","high"]],
    edgecolor="white"
)
axes[2].set_title("Category split by Severity")
axes[2].set_ylabel("Message count")
axes[2].set_xlabel("")
axes[2].legend(title="Severity", loc="upper right")
axes[2].set_xticklabels(CAT_ORDER, rotation=0)

plt.tight_layout()
plt.savefig(f"{OUT}/01_overview.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 2. MESSAGE-LEVEL TEXT STATISTICS
# ══════════════════════════════════════════════════════════════════════
print("[2/8] Message-level text statistics...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("2 · Message-Level Text Statistics by Category",
             fontsize=14, fontweight="bold", y=1.01)

metrics = [
    ("char_count",   "Character count",    "Characters"),
    ("word_count",   "Word count",          "Words"),
    ("exclamations", "Exclamation marks",   "Count"),
    ("caps_ratio",   "CAPS ratio",          "Fraction"),
    ("emoji_count",  "Emoji count",         "Count"),
    ("n_off_kw",     "Offensive keywords",  "Count"),
]

for ax, (col, title, ylabel) in zip(axes.flat, metrics):
    data_by_cat = [df[df["category"] == c][col].values for c in CAT_ORDER]
    bp = ax.boxplot(data_by_cat, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, cat in zip(bp["boxes"], CAT_ORDER):
        patch.set_facecolor(COLORS[cat])
        patch.set_alpha(0.75)
    ax.set_xticklabels(CAT_ORDER, rotation=15)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)

plt.tight_layout()
plt.savefig(f"{OUT}/02_text_stats.png", bbox_inches="tight")
plt.close()

# Print summary table
print("\n  Mean values by category:")
summary = df.groupby("category")[
    ["char_count","word_count","exclamations","caps_ratio","n_off_kw","n_gam_kw"]
].mean().round(2).reindex(CAT_ORDER)
print(summary.to_string())


# ══════════════════════════════════════════════════════════════════════
# 3. VOCABULARY & TOP WORDS PER CATEGORY
# ══════════════════════════════════════════════════════════════════════
print("\n[3/8] Vocabulary analysis...")

# Stop words to exclude (common Danish words that carry no signal)
STOPWORDS = {
    "og","er","det","jeg","har","ikke","at","en","til","den","de",
    "med","et","på","for","af","som","der","i","her","så","vi",
    "men","man","kan","hun","han","var","om","da","nu","hvad",
    "nogen","godt","haha","lol","hej","jo","nå","ej","ja","nej",
    "okay","bare","lidt","mere","lige","alt","alle","også",
}

def top_words(subset_df, n=15):
    """Get the top N words from messages in a subset, excluding stopwords."""
    all_words = []
    for msg in subset_df["content"].str.lower():
        words = re.findall(r"[a-zæøå]{3,}", msg)  # 3+ letter Danish words
        all_words.extend([w for w in words if w not in STOPWORDS])
    return Counter(all_words).most_common(n)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("3 · Top Words per Category (stopwords removed)",
             fontsize=14, fontweight="bold", y=1.01)

for ax, cat in zip(axes.flat, CAT_ORDER):
    words = top_words(df[df["category"] == cat])
    if not words:
        continue
    labels, counts = zip(*words)
    bars = ax.barh(labels[::-1], counts[::-1], color=COLORS[cat], alpha=0.85)
    ax.set_title(f'"{cat}" — top words', fontweight="bold")
    ax.set_xlabel("Frequency")
    for bar, n in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(n), va="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/03_top_words.png", bbox_inches="tight")
plt.close()

# 3b: Word clouds
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle("3b · Word Clouds per Category",
             fontsize=13, fontweight="bold")

for ax, cat in zip(axes, CAT_ORDER):
    text = " ".join(df[df["category"] == cat]["content"].str.lower())
    # Remove stopwords and short words
    clean_text = " ".join(
        w for w in re.findall(r"[a-zæøå]{3,}", text) if w not in STOPWORDS
    )
    if clean_text.strip():
        wc = WordCloud(
            width=400, height=300, background_color="white",
            colormap="RdYlGn" if cat == "clean" else
                     "Reds"  if cat == "offensive" else
                     "Oranges" if cat == "gambling" else "Purples",
            max_words=60, collocations=False,
        ).generate(clean_text)
        ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(cat, fontweight="bold", color=COLORS[cat])

plt.tight_layout()
plt.savefig(f"{OUT}/03b_wordclouds.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 4. USER-LEVEL BEHAVIOR PROFILES
# ══════════════════════════════════════════════════════════════════════
print("[4/8] User behavior profiles...")

user_stats = df.groupby("user_id").agg(
    total_msgs     = ("msg_id",      "count"),
    flagged_msgs   = ("is_flagged",  "sum"),
    off_kw_total   = ("n_off_kw",    "sum"),
    gam_kw_total   = ("n_gam_kw",    "sum"),
    avg_char       = ("char_count",  "mean"),
    avg_excl       = ("exclamations","mean"),
    has_high_sev   = ("severity",    lambda x: (x == "high").any()),
).reset_index()

user_stats["flag_rate"]  = user_stats["flagged_msgs"] / user_stats["total_msgs"]
user_stats["dominant_cat"] = df.groupby("user_id").apply(
    lambda g: g[g["category"] != "clean"]["category"].mode().iloc[0]
    if (g["category"] != "clean").any() else "clean"
).values

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("4 · User-Level Behavior Profiles",
             fontsize=14, fontweight="bold", y=1.01)

# 4a: Messages per user, colored by dominant category
user_stats_sorted = user_stats.sort_values("total_msgs")
colors_u = [COLORS[c] for c in user_stats_sorted["dominant_cat"]]
axes[0, 0].barh(
    user_stats_sorted["user_id"].astype(str),
    user_stats_sorted["total_msgs"],
    color=colors_u, edgecolor="white"
)
axes[0, 0].set_title("Messages per User (color = dominant category)",
                     fontweight="bold")
axes[0, 0].set_xlabel("Message count")
axes[0, 0].set_ylabel("User ID")
legend_patches = [mpatches.Patch(color=COLORS[c], label=c) for c in CAT_ORDER]
axes[0, 0].legend(handles=legend_patches, loc="lower right", fontsize=8)

# 4b: Flag rate per user
user_stats_sorted2 = user_stats[user_stats["flag_rate"] > 0].sort_values("flag_rate")
bar_colors = [COLORS[c] for c in user_stats_sorted2["dominant_cat"]]
axes[0, 1].barh(
    user_stats_sorted2["user_id"].astype(str),
    user_stats_sorted2["flag_rate"],
    color=bar_colors, edgecolor="white"
)
axes[0, 1].set_title("Flag Rate per User (flagged msgs / total msgs)",
                     fontweight="bold")
axes[0, 1].set_xlabel("Proportion of flagged messages")
axes[0, 1].set_ylabel("User ID")
axes[0, 1].axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5,
                   label="50% threshold")
axes[0, 1].legend(fontsize=8)

# 4c: Offensive vs Gambling keyword totals per user (scatter)
sc = axes[1, 0].scatter(
    user_stats["off_kw_total"], user_stats["gam_kw_total"],
    c=[COLORS[c] for c in user_stats["dominant_cat"]],
    s=user_stats["total_msgs"] * 2,  # bubble size = message volume
    alpha=0.75, edgecolors="white", linewidths=0.5
)
axes[1, 0].set_title("Offensive vs Gambling Keywords per User\n"
                      "(bubble size = total messages)", fontweight="bold")
axes[1, 0].set_xlabel("Total offensive keywords used")
axes[1, 0].set_ylabel("Total gambling keywords used")
axes[1, 0].legend(handles=legend_patches, fontsize=8)

# Annotate interesting users
for _, row in user_stats[user_stats["off_kw_total"] > 8].iterrows():
    axes[1, 0].annotate(f"u{int(row['user_id'])}",
                        (row["off_kw_total"], row["gam_kw_total"]),
                        fontsize=7, ha="center")
for _, row in user_stats[user_stats["gam_kw_total"] > 8].iterrows():
    axes[1, 0].annotate(f"u{int(row['user_id'])}",
                        (row["off_kw_total"], row["gam_kw_total"]),
                        fontsize=7, ha="center")

# 4d: User risk matrix — how many users fall into each category/severity combo
risk_df = df[df["category"] != "clean"].groupby(
    ["user_id", "category", "severity"]
).size().reset_index(name="msg_count")
risk_pivot = risk_df.pivot_table(
    index="category", columns="severity",
    values="user_id", aggfunc=pd.Series.nunique, fill_value=0
).reindex(["offensive","gambling","both"]).reindex(
    columns=["low","medium","high"], fill_value=0
)
sns.heatmap(risk_pivot, annot=True, fmt="d", cmap="YlOrRd",
            ax=axes[1, 1], linewidths=0.5, cbar_kws={"label": "Unique users"})
axes[1, 1].set_title("Unique Flagged Users per Category × Severity",
                     fontweight="bold")
axes[1, 1].set_xlabel("Severity")
axes[1, 1].set_ylabel("Category")

plt.tight_layout()
plt.savefig(f"{OUT}/04_user_profiles.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 5. TEMPORAL PATTERNS
# ══════════════════════════════════════════════════════════════════════
print("[5/8] Temporal patterns...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("5 · Temporal Patterns",
             fontsize=14, fontweight="bold", y=1.01)

# 5a: Messages by hour, stacked by category
hourly = df.groupby(["hour", "category"]).size().unstack(fill_value=0)
hourly = hourly.reindex(columns=CAT_ORDER, fill_value=0)
hourly.plot(kind="bar", stacked=True, ax=axes[0, 0],
            color=[COLORS[c] for c in CAT_ORDER], edgecolor="none", width=0.8)
axes[0, 0].set_title("Messages by Hour of Day", fontweight="bold")
axes[0, 0].set_xlabel("Hour")
axes[0, 0].set_ylabel("Message count")
axes[0, 0].set_xticklabels(hourly.index, rotation=0, fontsize=8)
axes[0, 0].legend(title="Category", fontsize=8)

# 5b: Flag rate by hour (proportion of messages that are problematic)
hourly_flag = df.groupby("hour").agg(
    total=("msg_id","count"), flagged=("is_flagged","sum")
)
hourly_flag["flag_rate"] = hourly_flag["flagged"] / hourly_flag["total"]
axes[0, 1].plot(hourly_flag.index, hourly_flag["flag_rate"],
                marker="o", color="#e74c3c", linewidth=2.5)
axes[0, 1].fill_between(hourly_flag.index, hourly_flag["flag_rate"],
                         alpha=0.2, color="#e74c3c")
axes[0, 1].set_title("Flag Rate by Hour of Day", fontweight="bold")
axes[0, 1].set_xlabel("Hour")
axes[0, 1].set_ylabel("Proportion flagged")
axes[0, 1].set_ylim(0, 1)
axes[0, 1].axhline(df["is_flagged"].mean(), color="gray", linestyle="--",
                    linewidth=1, label=f"Overall avg ({df['is_flagged'].mean():.2f})")
axes[0, 1].legend(fontsize=8)

# 5c: Messages by weekday
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_counts = df.groupby(["weekday","category"]).size().unstack(fill_value=0)
weekday_counts = weekday_counts.reindex(
    [w for w in weekday_order if w in weekday_counts.index]
).reindex(columns=CAT_ORDER, fill_value=0)
weekday_counts.plot(kind="bar", stacked=True, ax=axes[1, 0],
                    color=[COLORS[c] for c in CAT_ORDER], edgecolor="none")
axes[1, 0].set_title("Messages by Day of Week", fontweight="bold")
axes[1, 0].set_xlabel("")
axes[1, 0].set_ylabel("Message count")
axes[1, 0].set_xticklabels(weekday_counts.index, rotation=25, ha="right", fontsize=9)
axes[1, 0].legend(title="Category", fontsize=8)

# 5d: Cumulative flagged messages over time
df_sorted = df.sort_values("timestamp")
df_sorted["cumulative_flagged"] = df_sorted["is_flagged"].cumsum()
df_sorted["cumulative_total"]   = range(1, len(df_sorted) + 1)
axes[1, 1].plot(df_sorted["cumulative_total"], df_sorted["cumulative_flagged"],
                color="#e74c3c", linewidth=2, label="Flagged messages")
axes[1, 1].plot(df_sorted["cumulative_total"], df_sorted["cumulative_total"],
                color="#bdc3c7", linewidth=1, linestyle="--", label="If all were flagged")
axes[1, 1].set_title("Cumulative Flagged Messages Over Time", fontweight="bold")
axes[1, 1].set_xlabel("Total messages seen")
axes[1, 1].set_ylabel("Cumulative flagged")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/05_temporal.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 6. ESCALATION & SEVERITY PROGRESSION
# ══════════════════════════════════════════════════════════════════════
print("[6/8] Escalation & severity progression...")

# Map severity to numbers for trend analysis
sev_num = {"none": 0, "low": 1, "medium": 2, "high": 3}
df["severity_num"] = df["severity"].map(sev_num)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("6 · Escalation & Severity Progression",
             fontsize=14, fontweight="bold", y=1.01)

# 6a: Average severity score by message position in user's chat history
# (Does severity increase as a user stays in the chat?)
df["msg_rank_bin"] = pd.cut(df["msg_rank"], bins=5,
                             labels=["1st 20%","2nd 20%","3rd 20%","4th 20%","5th 20%"])
sev_by_rank = df[df["category"] != "clean"].groupby(
    "msg_rank_bin", observed=True
)["severity_num"].mean()
axes[0].bar(sev_by_rank.index, sev_by_rank.values,
            color=sns.color_palette("YlOrRd", 5), edgecolor="white")
axes[0].set_title("Avg Severity by Position in User History\n"
                   "(1st 20% = their earliest messages)", fontweight="bold")
axes[0].set_xlabel("Position in user's chat history")
axes[0].set_ylabel("Avg severity score (0=none, 3=high)")
axes[0].set_ylim(0, 3)

# 6b: First vs last severity for each problematic user
# Compare the severity of a user's first flagged message vs their last
flagged_users = df[df["is_flagged"]].groupby("user_id")
first_sev = flagged_users["severity_num"].first()
last_sev  = flagged_users["severity_num"].last()
escalation_df = pd.DataFrame({"first": first_sev, "last": last_sev}).dropna()
escalation_df["delta"] = escalation_df["last"] - escalation_df["first"]

colors_esc = ["#c0392b" if d > 0 else "#2ecc71" if d < 0 else "#bdc3c7"
              for d in escalation_df["delta"]]
axes[1].bar(escalation_df.index.astype(str), escalation_df["delta"],
            color=colors_esc, edgecolor="white")
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_title("Severity Change: First → Last Flagged Message\n"
                   "(red = escalated, green = de-escalated)", fontweight="bold")
axes[1].set_xlabel("User ID")
axes[1].set_ylabel("Severity delta")
axes[1].set_xticklabels(escalation_df.index.astype(str), rotation=45, fontsize=7)

escalated   = (escalation_df["delta"] > 0).sum()
deescalated = (escalation_df["delta"] < 0).sum()
stable      = (escalation_df["delta"] == 0).sum()
axes[1].text(0.02, 0.97,
             f"Escalated: {escalated}\nDe-escalated: {deescalated}\nStable: {stable}",
             transform=axes[1].transAxes, va="top", fontsize=9,
             bbox={"boxstyle":"round","facecolor":"white","alpha":0.8})

# 6c: Transition heatmap — for each user, how does category change over time?
# Split each user's messages into first half vs second half, compare distribution
def half_category(group):
    mid = len(group) // 2
    first_half = group.iloc[:mid]["category"].value_counts(normalize=True)
    last_half  = group.iloc[mid:]["category"].value_counts(normalize=True)
    return pd.Series({
        "clean_first":     first_half.get("clean",     0),
        "clean_last":      last_half.get("clean",      0),
        "offensive_first": first_half.get("offensive", 0),
        "offensive_last":  last_half.get("offensive",  0),
        "gambling_first":  first_half.get("gambling",  0),
        "gambling_last":   last_half.get("gambling",   0),
    })

halves = df.groupby("user_id").apply(half_category)
transition_data = pd.DataFrame({
    "Category": ["Clean", "Offensive", "Gambling"],
    "First half": [halves["clean_first"].mean(), halves["offensive_first"].mean(),
                   halves["gambling_first"].mean()],
    "Second half": [halves["clean_last"].mean(), halves["offensive_last"].mean(),
                    halves["gambling_last"].mean()],
})
x = np.arange(3)
w = 0.35
axes[2].bar(x - w/2, transition_data["First half"],  w, label="First half of history",
            color=["#2ecc71","#e74c3c","#f39c12"], alpha=0.6)
axes[2].bar(x + w/2, transition_data["Second half"], w, label="Second half of history",
            color=["#2ecc71","#e74c3c","#f39c12"], alpha=1.0)
axes[2].set_xticks(x)
axes[2].set_xticklabels(["Clean", "Offensive", "Gambling"])
axes[2].set_title("Avg Category Distribution:\nFirst vs Second Half of User History",
                  fontweight="bold")
axes[2].set_ylabel("Average proportion")
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/06_escalation.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 7. EDGE CASES & AMBIGUITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("[7/8] Edge case & ambiguity analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("7 · Edge Cases & Ambiguity", fontsize=14, fontweight="bold", y=1.01)

# 7a: Clean messages that contain offensive keywords (potential false positives)
clean_with_off_kw = df[(df["category"] == "clean") & (df["n_off_kw"] > 0)]
clean_with_gam_kw = df[(df["category"] == "clean") & (df["n_gam_kw"] > 0)]

edge_summary = {
    "Clean with\noffensive\nkeywords": len(clean_with_off_kw),
    "Clean with\ngambling\nkeywords":  len(clean_with_gam_kw),
    "Offensive\nwith gambling\nkeywords": len(
        df[(df["category"] == "offensive") & (df["n_gam_kw"] > 0)]),
    "Gambling\nwith offensive\nkeywords": len(
        df[(df["category"] == "gambling") & (df["n_off_kw"] > 0)]),
}
axes[0, 0].bar(edge_summary.keys(), edge_summary.values(),
               color=["#3498db","#27ae60","#8e44ad","#c0392b"], edgecolor="white")
axes[0, 0].set_title("Cross-Signal Contamination\n(Messages where signals conflict)",
                     fontweight="bold")
axes[0, 0].set_ylabel("Message count")
for i, (k, v) in enumerate(edge_summary.items()):
    axes[0, 0].text(i, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")

# 7b: Message length distribution for clean vs ambiguous messages
axes[0, 1].hist(df[df["category"] == "clean"]["char_count"],
                bins=30, alpha=0.6, label="Clean", color="#2ecc71", density=True)
axes[0, 1].hist(df[df["is_flagged"]]["char_count"],
                bins=30, alpha=0.6, label="Flagged", color="#e74c3c", density=True)
axes[0, 1].hist(clean_with_off_kw["char_count"],
                bins=20, alpha=0.8, label="Clean + offensive kw (edge)",
                color="#3498db", density=True, linestyle="--",
                histtype="step", linewidth=2)
axes[0, 1].set_title("Message Length: Clean vs Flagged vs Edge Cases",
                     fontweight="bold")
axes[0, 1].set_xlabel("Character count")
axes[0, 1].set_ylabel("Density")
axes[0, 1].legend(fontsize=8)

# 7c: Keyword co-occurrence — does offensive_kw and gambling_kw ever appear together?
axes[1, 0].scatter(
    df["n_off_kw"] + np.random.uniform(-0.2, 0.2, len(df)),  # jitter for visibility
    df["n_gam_kw"] + np.random.uniform(-0.2, 0.2, len(df)),
    c=[COLORS[c] for c in df["category"]],
    alpha=0.4, s=20
)
axes[1, 0].set_title("Keyword Co-occurrence per Message\n"
                      "(offensive kw vs gambling kw)", fontweight="bold")
axes[1, 0].set_xlabel("# Offensive keywords")
axes[1, 0].set_ylabel("# Gambling keywords")
axes[1, 0].legend(handles=legend_patches, fontsize=8)

# 7d: Print actual edge case examples
axes[1, 1].axis("off")
examples = []
for _, row in clean_with_off_kw.head(6).iterrows():
    examples.append(f'✓ CLEAN but has offensive kw:\n  "{row["content"][:65]}"')
for _, row in clean_with_gam_kw.head(4).iterrows():
    examples.append(f'✓ CLEAN but has gambling kw:\n  "{row["content"][:65]}"')

text_content = "\n\n".join(examples)
axes[1, 1].text(0.02, 0.98, "Edge Case Examples\n" + "─" * 45 + "\n\n" + text_content,
                transform=axes[1, 1].transAxes, va="top", fontsize=8,
                fontfamily="monospace",
                bbox={"boxstyle":"round","facecolor":"#f8f9fa","alpha":0.9})

plt.tight_layout()
plt.savefig(f"{OUT}/07_edge_cases.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 8. ACTIONABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("[8/8] Actionability analysis...")

ACTION_TABLE = {
    ("clean",     "none"):   "Ingen handling",
    ("offensive", "low"):    "Send advarsel i chatforum",
    ("offensive", "medium"): "Opret sag i backoffice",
    ("offensive", "high"):   "Blokér brugeren fra chat",
    ("gambling",  "low"):    "Send mail om ansvarligt spil",
    ("gambling",  "medium"): "Opret sag i backoffice",
    ("gambling",  "high"):   "Sæt spilgrænser / selvudelukkelse",
    ("both",      "low"):    "Send advarsel + mail",
    ("both",      "medium"): "Opret sag i backoffice (begge)",
    ("both",      "high"):   "Blokér + Sæt spilgrænser",
}
df["action"] = df.apply(
    lambda r: ACTION_TABLE.get((r["category"], r["severity"]), "Opret sag"),
    axis=1
)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("8 · Actionability Analysis",
             fontsize=14, fontweight="bold", y=1.01)

# 8a: Distribution of actions triggered
action_counts = df["action"].value_counts()
action_colors = {
    "Ingen handling":                 "#2ecc71",
    "Send advarsel i chatforum":      "#f1c40f",
    "Send mail om ansvarligt spil":   "#f39c12",
    "Send advarsel + mail":           "#e67e22",
    "Opret sag i backoffice":         "#e74c3c",
    "Opret sag i backoffice (begge)": "#c0392b",
    "Blokér brugeren fra chat":       "#8e44ad",
    "Sæt spilgrænser / selvudelukkelse": "#6c3483",
    "Blokér + Sæt spilgrænser":       "#2c3e50",
    "Opret sag":                      "#e74c3c",
}
colors_action = [action_colors.get(a, "#999") for a in action_counts.index]
bars = axes[0].barh(action_counts.index, action_counts.values,
                    color=colors_action, edgecolor="white")
axes[0].set_title("Actions Triggered\n(message count)", fontweight="bold")
axes[0].set_xlabel("Number of messages")
for bar, n in zip(bars, action_counts.values):
    axes[0].text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                 str(n), va="center", fontsize=8)

# 8b: Unique users affected per action
user_action = df[df["category"] != "clean"].groupby("user_id")["action"].value_counts()
users_per_action = df[df["action"] != "Ingen handling"].groupby("action")["user_id"].nunique()
users_per_action_sorted = users_per_action.sort_values()
colors_upa = [action_colors.get(a, "#999") for a in users_per_action_sorted.index]
bars2 = axes[1].barh(users_per_action_sorted.index, users_per_action_sorted.values,
                     color=colors_upa, edgecolor="white")
axes[1].set_title("Unique Users Affected\nper Action Type", fontweight="bold")
axes[1].set_xlabel("Number of unique users")
for bar, n in zip(bars2, users_per_action_sorted.values):
    axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(n), va="center", fontsize=8)

# 8c: Severity pyramid — how many messages fall into each tier
severity_flagged = df[df["category"] != "clean"]["severity"].value_counts()
tier_data = {
    "HIGH\n(immediate action)": severity_flagged.get("high", 0),
    "MEDIUM\n(backoffice case)": severity_flagged.get("medium", 0),
    "LOW\n(automated warning)": severity_flagged.get("low", 0),
}
bars3 = axes[2].bar(tier_data.keys(), tier_data.values(),
                    color=["#c0392b", "#e67e22", "#f1c40f"],
                    edgecolor="white", width=0.5)
axes[2].set_title("Severity Pyramid\n(flagged messages only)", fontweight="bold")
axes[2].set_ylabel("Message count")
for bar, (tier, n) in zip(bars3, tier_data.items()):
    pct = n / len(df[df["category"] != "clean"]) * 100
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2, f"{n}\n({pct:.1f}%)",
                 ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/08_actionability.png", bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA COMPLETE — KEY FINDINGS")
print("=" * 60)

total     = len(df)
flagged   = df["is_flagged"].sum()
pct_clean = len(df[df["category"] == "clean"]) / total * 100
pct_flag  = flagged / total * 100

print(f"\nDataset:          {total} messages | {df['user_id'].nunique()} users")
print(f"Clean messages:   {int(total * pct_clean / 100)} ({pct_clean:.1f}%)")
print(f"Flagged messages: {flagged} ({pct_flag:.1f}%)")
print(f"\nText characteristics:")
print(f"  Avg message length (clean):     {df[df['category']=='clean']['char_count'].mean():.1f} chars")
print(f"  Avg message length (offensive): {df[df['category']=='offensive']['char_count'].mean():.1f} chars")
print(f"  Avg message length (gambling):  {df[df['category']=='gambling']['char_count'].mean():.1f} chars")
print(f"\nEdge cases:")
print(f"  Clean msgs with offensive keywords: {len(clean_with_off_kw)}")
print(f"  Clean msgs with gambling keywords:  {len(clean_with_gam_kw)}")
print(f"\nEscalation:")
print(f"  Users who escalated (got worse):    {escalated}")
print(f"  Users who de-escalated:             {deescalated}")
print(f"  Users who stayed stable:            {stable}")
print(f"\nActions triggered (unique messages):")
for action, count in action_counts.items():
    if action != "Ingen handling":
        print(f"  {action[:45]:45s} {count}")

print(f"\nAll plots saved to {OUT}/")
print("Done! ✓")
