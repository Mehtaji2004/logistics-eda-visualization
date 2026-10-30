"""
Week 3 Task: Advanced Data Analysis and Visualization in Logistics
===================================================================
This script simulates a hypothetical logistics dataset of 2,000 shipments,
performs comprehensive Exploratory Data Analysis (EDA), and generates
professional-quality visualizations. Each chart is justified by its ability
to reveal distribution, trend, correlation, or comparison insights relevant
to logistics performance metrics.

Author  : Akshit Mehta
Date    : October 2026
Python  : 3.10+
Packages: pandas, numpy, matplotlib, seaborn, scipy

Run:
    python analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
#  1. DATA SIMULATION — Hypothetical Logistics Dataset
# ============================================================================
#  A reproducible synthetic dataset with key logistics variables:
#  distance, shipment volume, delivery time, transportation cost,
#  on-time flag, carrier, transport mode, region, and month.
# ============================================================================

rng = np.random.default_rng(42)             # fixed seed → reproducible results
N = 2000                                    # number of shipment records

# Categorical dimensions
regions  = ["North", "South", "East", "West", "Central"]
carriers = ["FastTrack", "SwiftMove", "EcoFreight", "PrimeCargo"]
modes    = ["Road", "Rail", "Air", "Sea"]

# Month names for readability
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

df = pd.DataFrame({
    "shipment_id":       [f"SH{i:05d}" for i in range(1, N + 1)],
    "region":            rng.choice(regions, N, p=[0.22, 0.20, 0.20, 0.20, 0.18]),
    "carrier":           rng.choice(carriers, N),
    "transport_mode":    rng.choice(modes, N, p=[0.55, 0.20, 0.15, 0.10]),
    "distance_km":       rng.gamma(shape=3.0, scale=250, size=N).round(1),
    "shipment_volume_kg": rng.lognormal(mean=6.0, sigma=0.7, size=N).round(1),
})
df["month"] = rng.integers(1, 13, N)

# Delivery time (hours) — depends on distance, transport mode, seasonal peak,
# and random noise.  Air is by far the fastest; sea is the slowest.
speed_map = {"Road": 55, "Rail": 45, "Air": 600, "Sea": 30}
speed = df["transport_mode"].map(speed_map)
base_hours = df["distance_km"] / speed + 4                          # base transit
peak_season = (
    df["month"].isin([10, 11, 12]).astype(int)
    * rng.normal(6, 2, N).clip(0)
)
df["delivery_time_hrs"] = (
    base_hours + peak_season + rng.normal(0, 3, N)
).clip(2).round(1)

# Transportation cost (USD) — driven by distance, volume, and per-km rate.
rate_map = {"Road": 0.9, "Rail": 0.6, "Air": 4.5, "Sea": 0.35}
rate = df["transport_mode"].map(rate_map)
df["transport_cost_usd"] = (
    50 + df["distance_km"] * rate + df["shipment_volume_kg"] * 0.15
    + rng.normal(0, 60, N)
).clip(30).round(2)

# On-time flag — logistic function: slower deliveries are more likely to be late.
late_prob = 1 / (1 + np.exp(-(df["delivery_time_hrs"] - 30) / 8))
df["on_time"] = (rng.random(N) > late_prob * 0.6).astype(int)

# Derived metric: cost efficiency
df["cost_per_kg"] = (df["transport_cost_usd"] / df["shipment_volume_kg"]).round(3)

# Persist the simulated dataset
os.makedirs("charts", exist_ok=True)
df.to_csv("logistics_data.csv", index=False)
print(f"✅  Dataset created: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ============================================================================
#  2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

# 2.1  Quick inspection
print("=" * 70)
print("SECTION 2.1 — DATASET OVERVIEW")
print("=" * 70)
print(f"\nShape   : {df.shape}")
print(f"Columns : {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}\n")
print(f"First 5 rows:\n{df.head()}\n")

# 2.2  Missing-value check
print("=" * 70)
print("SECTION 2.2 — MISSING VALUES")
print("=" * 70)
missing = df.isna().sum()
print(f"\n{missing}")
print(f"\nTotal missing: {missing.sum()}  →  Dataset is complete.\n")

# 2.3  Descriptive statistics — central tendency, spread, and shape
numeric_cols = [
    "distance_km", "shipment_volume_kg", "delivery_time_hrs",
    "transport_cost_usd", "cost_per_kg",
]

print("=" * 70)
print("SECTION 2.3 — DESCRIPTIVE STATISTICS")
print("=" * 70)
summary = df[numeric_cols].agg(["mean", "median", "std", "min", "max"]).T
summary["skew"]     = df[numeric_cols].skew().round(3)
summary["kurtosis"] = df[numeric_cols].kurtosis().round(3)
summary["IQR"]      = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
print(f"\n{summary.round(2)}\n")
summary.round(2).to_csv("summary_statistics.csv")

# 2.4  Correlation analysis
print("=" * 70)
print("SECTION 2.4 — CORRELATION MATRIX")
print("=" * 70)
corr = df[numeric_cols + ["on_time"]].corr()
print(f"\n{corr.round(2)}\n")
corr.round(2).to_csv("correlation_matrix.csv")

# 2.5  Group-level aggregations
print("=" * 70)
print("SECTION 2.5 — GROUP COMPARISONS")
print("=" * 70)

# By transport mode
by_mode = df.groupby("transport_mode").agg(
    shipments        = ("shipment_id",       "count"),
    avg_distance     = ("distance_km",       "mean"),
    avg_delivery_hrs = ("delivery_time_hrs",  "mean"),
    avg_cost         = ("transport_cost_usd", "mean"),
    avg_cost_per_kg  = ("cost_per_kg",        "mean"),
    on_time_rate     = ("on_time",            "mean"),
).round(3)
print(f"\n--- By Transport Mode ---\n{by_mode}\n")
by_mode.to_csv("by_mode.csv")

# By carrier
by_carrier = df.groupby("carrier").agg(
    shipments        = ("shipment_id",       "count"),
    avg_distance     = ("distance_km",       "mean"),
    avg_delivery_hrs = ("delivery_time_hrs",  "mean"),
    avg_cost         = ("transport_cost_usd", "mean"),
    avg_cost_per_kg  = ("cost_per_kg",        "mean"),
    on_time_rate     = ("on_time",            "mean"),
).round(3)
print(f"--- By Carrier ---\n{by_carrier}\n")
by_carrier.to_csv("by_carrier.csv")

# By region
by_region = df.groupby("region").agg(
    shipments        = ("shipment_id",       "count"),
    avg_distance     = ("distance_km",       "mean"),
    avg_delivery_hrs = ("delivery_time_hrs",  "mean"),
    avg_cost         = ("transport_cost_usd", "mean"),
    avg_cost_per_kg  = ("cost_per_kg",        "mean"),
    on_time_rate     = ("on_time",            "mean"),
).round(3)
print(f"--- By Region ---\n{by_region}\n")
by_region.to_csv("by_region.csv")

# Monthly trends
monthly = df.groupby("month").agg(
    shipments        = ("shipment_id",       "count"),
    avg_delivery_hrs = ("delivery_time_hrs",  "mean"),
    avg_cost         = ("transport_cost_usd", "mean"),
    on_time_rate     = ("on_time",            "mean"),
).round(3)
monthly.to_csv("monthly.csv")

# 2.6  Outlier detection (IQR method)
print("=" * 70)
print("SECTION 2.6 — OUTLIER DETECTION (IQR METHOD)")
print("=" * 70)
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    pct = n_outliers / len(df) * 100
    print(f"  {col:25s}  outliers = {n_outliers:4d}  ({pct:.1f}%)  "
          f"bounds = [{lower:.1f}, {upper:.1f}]")

# 2.7  Statistical tests — mode-level delivery time differences
print(f"\n{'=' * 70}")
print("SECTION 2.7 — KRUSKAL-WALLIS TEST (Delivery Time ~ Transport Mode)")
print("=" * 70)
groups = [g["delivery_time_hrs"].values for _, g in df.groupby("transport_mode")]
stat, p_val = stats.kruskal(*groups)
print(f"  H-statistic = {stat:.2f},  p-value = {p_val:.2e}")
print(f"  → {'Significant' if p_val < 0.05 else 'Not significant'} difference "
      f"in delivery times across transport modes (α = 0.05).\n")

# ============================================================================
#  3. VISUALIZATIONS — 10 Professional Charts
# ============================================================================
#  Theme: clean whitegrid with a cohesive colour palette.
#  Every chart is saved at 200 dpi for crisp DOC embedding.
# ============================================================================

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
DPI = 200
FIG_DIR = "charts"

# ---- Helper ----------------------------------------------------------------
def save(fig, name: str) -> None:
    """Save figure, close it, and print confirmation."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Saved {path}")


print("=" * 70)
print("SECTION 3 — GENERATING VISUALIZATIONS")
print("=" * 70)

# ---- CHART 1: Delivery-Time Distribution ----------------------------------
#  WHY: A histogram with KDE reveals the shape, centre, and spread of delivery
#  times.  Mean vs median comparison indicates skewness.
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.histplot(df["delivery_time_hrs"], bins=45, kde=True, ax=ax,
             color="#2a6f97", edgecolor="white", linewidth=0.4)
mean_val = df["delivery_time_hrs"].mean()
med_val  = df["delivery_time_hrs"].median()
ax.axvline(mean_val, color="#e63946", ls="--", lw=1.8,
           label=f"Mean = {mean_val:.1f} hrs")
ax.axvline(med_val,  color="#2d6a4f", ls="--", lw=1.8,
           label=f"Median = {med_val:.1f} hrs")
ax.set_title("Distribution of Delivery Time", fontsize=14, weight="bold")
ax.set_xlabel("Delivery Time (hours)")
ax.set_ylabel("Number of Shipments")
ax.legend(frameon=True, fontsize=9)
save(fig, "01_delivery_time_distribution.png")

# ---- CHART 2: Cost Box-Plot by Transport Mode ------------------------------
#  WHY: Box plots compare medians, spreads, and outliers across categories.
#  Air clearly dominates cost; sea is cheapest.
fig, ax = plt.subplots(figsize=(9, 5.5))
order = ["Sea", "Rail", "Road", "Air"]
palette = {"Sea": "#219ebc", "Rail": "#8ecae6", "Road": "#fb8500", "Air": "#e63946"}
sns.boxplot(data=df, x="transport_mode", y="transport_cost_usd",
            order=order, palette=palette, ax=ax, linewidth=1.2, fliersize=3)
ax.set_title("Transportation Cost by Transport Mode", fontsize=14, weight="bold")
ax.set_xlabel("Transport Mode")
ax.set_ylabel("Cost (USD)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
save(fig, "02_cost_by_mode_boxplot.png")

# ---- CHART 3: Distance vs Cost Scatter (colour = mode) --------------------
#  WHY: Scatter plots visualise pairwise relationships.  Colour-coding by mode
#  shows how per-km rates differ.
sample = df.sample(900, random_state=1)
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.scatterplot(data=sample, x="distance_km", y="transport_cost_usd",
                hue="transport_mode", style="transport_mode",
                palette=palette, alpha=0.65, s=35, ax=ax)
ax.set_title("Distance vs Transportation Cost", fontsize=14, weight="bold")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Cost (USD)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
ax.legend(title="Mode", frameon=True, fontsize=9, title_fontsize=10)
save(fig, "03_distance_vs_cost_scatter.png")

# ---- CHART 4: Correlation Heatmap ------------------------------------------
#  WHY: A heatmap summarises all pairwise correlations in one view, helping
#  identify the strongest drivers (e.g., distance → cost, distance → time).
fig, ax = plt.subplots(figsize=(8, 6.5))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            mask=mask, square=True, linewidths=0.8,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"}, ax=ax)
ax.set_title("Correlation Heatmap of Key Variables", fontsize=14, weight="bold")
save(fig, "04_correlation_heatmap.png")

# ---- CHART 5: Monthly Trend — Delivery Time & On-Time Rate ----------------
#  WHY: A dual-axis line chart reveals seasonal patterns.  Peak-season months
#  (Oct–Dec) show higher delivery times and lower on-time rates.
fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax2 = ax1.twinx()

colour1, colour2 = "#bc4749", "#2a9d8f"
ax1.plot(monthly.index, monthly["avg_delivery_hrs"], marker="o", color=colour1,
         lw=2, label="Avg Delivery Time")
ax1.fill_between(monthly.index, monthly["avg_delivery_hrs"], alpha=0.08,
                 color=colour1)
ax2.plot(monthly.index, monthly["on_time_rate"] * 100, marker="s", color=colour2,
         lw=2, ls="--", label="On-Time Rate (%)")

ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(MONTH_LABELS)
ax1.set_xlabel("Month")
ax1.set_ylabel("Avg Delivery Time (hrs)", color=colour1)
ax2.set_ylabel("On-Time Rate (%)", color=colour2)
ax2.set_ylim(60, 100)
ax1.set_title("Seasonal Trend: Delivery Time & On-Time Rate",
              fontsize=14, weight="bold")

# Highlight peak season
for m in [10, 11, 12]:
    ax1.axvspan(m - 0.4, m + 0.4, alpha=0.07, color="grey")
ax1.annotate("Peak Season ▸", xy=(10, monthly.loc[10, "avg_delivery_hrs"]),
             fontsize=8, color="grey", ha="left",
             xytext=(9.2, monthly["avg_delivery_hrs"].max() + 1),
             arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
           frameon=True, fontsize=9)
save(fig, "05_monthly_trend.png")

# ---- CHART 6: On-Time Rate by Carrier (horizontal bar) --------------------
#  WHY: Horizontal bars make label reading easier when comparing carriers.
#  Percentage annotations add precision.
fig, ax = plt.subplots(figsize=(9, 5))
carrier_sorted = by_carrier.sort_values("on_time_rate")
colours = sns.color_palette("viridis", len(carrier_sorted))
bars = ax.barh(carrier_sorted.index, carrier_sorted["on_time_rate"] * 100,
               color=colours, edgecolor="white", height=0.55)
for bar, val in zip(bars, carrier_sorted["on_time_rate"] * 100):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10, weight="bold")
ax.set_xlim(0, 105)
ax.set_xlabel("On-Time Delivery Rate (%)")
ax.set_title("On-Time Delivery Rate by Carrier", fontsize=14, weight="bold")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
save(fig, "06_ontime_by_carrier.png")

# ---- CHART 7: Region × Mode Heatmap — Avg Delivery Time -------------------
#  WHY: A cross-tabulation heatmap highlights which region–mode combinations
#  are fastest or slowest, guiding route optimisation decisions.
pivot = df.pivot_table(
    values="delivery_time_hrs", index="region",
    columns="transport_mode", aggfunc="mean",
)
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.6,
            cbar_kws={"label": "Hours"}, ax=ax)
ax.set_title("Avg Delivery Time (hrs): Region × Transport Mode",
             fontsize=14, weight="bold")
ax.set_xlabel("Transport Mode")
ax.set_ylabel("Region")
save(fig, "07_region_mode_heatmap.png")

# ---- CHART 8: Violin Plot — Delivery Time by Region -----------------------
#  WHY: Violin plots combine a box plot with a KDE, showing the full shape
#  of each region's delivery-time distribution — not just quartiles.
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.violinplot(data=df, x="region", y="delivery_time_hrs",
               order=regions, inner="quartile", palette="Set2",
               linewidth=1, ax=ax)
ax.set_title("Delivery Time Distribution by Region", fontsize=14, weight="bold")
ax.set_xlabel("Region")
ax.set_ylabel("Delivery Time (hours)")
save(fig, "08_delivery_time_by_region_violin.png")

# ---- CHART 9: Pair Plot — Core Numeric Variables --------------------------
#  WHY: A pair plot exposes all pairwise scatter plots and marginal
#  distributions in one matrix, coloured by mode for quick comparison.
pair_sample = df[["distance_km", "delivery_time_hrs", "transport_cost_usd",
                   "transport_mode"]].sample(600, random_state=7)
g = sns.pairplot(pair_sample, hue="transport_mode", palette=palette,
                 plot_kws={"alpha": 0.5, "s": 20},
                 diag_kws={"alpha": 0.5}, height=2.4, aspect=1.0)
g.figure.suptitle("Pair Plot of Key Logistics Variables",
                  fontsize=14, weight="bold", y=1.02)
save(g.figure, "09_pairplot_key_variables.png")

# ---- CHART 10: Stacked Bar — Shipment Volume Share by Mode per Region -----
#  WHY: A percentage-stacked bar chart shows the modal mix in each region,
#  useful for strategic logistics planning and capacity allocation.
ct = pd.crosstab(df["region"], df["transport_mode"], normalize="index") * 100
ct = ct[["Road", "Rail", "Air", "Sea"]]     # consistent order
fig, ax = plt.subplots(figsize=(9, 5.5))
ct.plot(kind="bar", stacked=True, ax=ax,
        color=[palette[m] for m in ct.columns], edgecolor="white", linewidth=0.5)
ax.set_title("Transport Mode Mix by Region (%)", fontsize=14, weight="bold")
ax.set_xlabel("Region")
ax.set_ylabel("Share (%)")
ax.set_xticklabels(ct.index, rotation=0)
ax.legend(title="Mode", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
save(fig, "10_mode_mix_by_region_stacked.png")

# ============================================================================
#  4. SUMMARY OF KEY INSIGHTS
# ============================================================================

print(f"\n{'=' * 70}")
print("SECTION 4 — KEY ANALYTICAL INSIGHTS")
print("=" * 70)
insights = [
    f"1. Distance is strongly correlated with delivery time (r = {corr.loc['distance_km', 'delivery_time_hrs']:.2f}) "
    f"and cost (r = {corr.loc['distance_km', 'transport_cost_usd']:.2f}).",
    f"2. Air freight is the fastest but most expensive mode — avg cost "
    f"${by_mode.loc['Air', 'avg_cost']:,.0f} vs ${by_mode.loc['Sea', 'avg_cost']:,.0f} for sea.",
    f"3. Sea freight has the lowest on-time rate ({by_mode.loc['Sea', 'on_time_rate'] * 100:.1f}%) "
    f"vs air ({by_mode.loc['Air', 'on_time_rate'] * 100:.1f}%).",
    f"4. Peak-season months (Oct–Dec) see delivery times rise from ~{monthly.loc[6, 'avg_delivery_hrs']:.0f} hrs "
    f"to ~{monthly.loc[11, 'avg_delivery_hrs']:.0f} hrs, a "
    f"{((monthly.loc[11, 'avg_delivery_hrs'] / monthly.loc[6, 'avg_delivery_hrs']) - 1) * 100:.0f}% increase.",
    f"5. Carriers perform similarly (on-time range {by_carrier['on_time_rate'].min() * 100:.1f}%–"
    f"{by_carrier['on_time_rate'].max() * 100:.1f}%), suggesting mode and distance dominate performance.",
    f"6. Cost-per-kg efficiency is best for sea (${df[df['transport_mode'] == 'Sea']['cost_per_kg'].mean():.2f}/kg) "
    f"and worst for air (${df[df['transport_mode'] == 'Air']['cost_per_kg'].mean():.2f}/kg).",
    "7. The Kruskal-Wallis test confirms that delivery-time differences across modes "
    f"are statistically significant (H = {stat:.1f}, p < 0.001).",
]
for line in insights:
    print(f"  {line}")

# ============================================================================
#  5. RECOMMENDATIONS
# ============================================================================

print(f"\n{'=' * 70}")
print("SECTION 5 — RECOMMENDATIONS")
print("=" * 70)
recommendations = [
    "1. Shift non-urgent long-distance shipments from Air to Rail to cut costs by ~80%"
    "   while keeping delivery within acceptable time windows.",
    "2. Increase buffer stock or safety-stock levels in Oct–Dec to offset the seasonal"
    "   delivery-time spike and protect customer SLAs.",
    "3. Negotiate volume-based contracts with carriers — carrier performance is nearly"
    "   identical, so cost should be the differentiator.",
    "4. Invest in real-time tracking for sea shipments (lowest on-time rate) to improve"
    "   visibility and allow proactive delay management.",
    "5. Monitor cost-per-kg trends monthly; use dashboards built from this analysis"
    "   framework for continuous improvement.",
]
for line in recommendations:
    print(f"  {line}")

print(f"\n{'=' * 70}")
print("✅  Analysis complete.  10 charts saved in '{FIG_DIR}/'.")
print(f"{'=' * 70}\n")
