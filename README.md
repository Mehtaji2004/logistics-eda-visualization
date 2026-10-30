# Advanced Data Analysis and Visualization in Logistics

> Week 3 Internship Task — Exploratory Data Analysis (EDA) and professional visualization of a simulated logistics dataset using Python.

## 📌 Objective

Deepen analytical skills by exploring, analysing, and visualising logistics data.  The project simulates 2,000 shipment records, performs thorough EDA, produces 10 publication-quality charts, and documents actionable insights for logistics decision-making.

## 🔍 What This Project Does

| Step | Description |
|------|-------------|
| **1. Data Simulation** | Generates 2,000 shipment records with realistic variables: region, carrier, transport mode, distance, volume, delivery time, cost, and on-time status. |
| **2. Exploratory Data Analysis** | Calculates descriptive statistics (mean, median, std, skew, kurtosis, IQR), correlation matrix, group-level aggregations (by mode, carrier, region, month), outlier detection (IQR method), and a Kruskal-Wallis statistical test. |
| **3. Visualizations** | Produces 10 charts — histogram, box plot, scatter plot, correlation heatmap, dual-axis line chart, bar chart, cross-tabulation heatmap, violin plot, pair plot, and stacked bar chart. |
| **4. Insights & Recommendations** | Derives 7 key findings and 5 actionable recommendations for logistics optimisation. |

## 📊 Key Findings

1. **Distance drives performance** — strongly correlated with delivery time (*r* = 0.70) and cost (*r* = 0.42).
2. **Air is fastest but most expensive** — avg cost \$3,516 vs \$406 for sea (an 8.7× premium).
3. **Sea freight is the least reliable** — 69% on-time vs 96% for air.
4. **Peak-season spike** — delivery time rises ~30% in Oct–Dec, impacting customer SLAs.
5. **Carriers perform uniformly** — on-time rates span only 82.8%–85.0%, indicating mode and distance dominate performance.
6. **Cost-per-kg efficiency** — sea offers the best cost-per-kg; air the worst.
7. **Statistical confirmation** — the Kruskal-Wallis test confirms delivery-time differences across modes are statistically significant (*p* < 0.001).

## 🗂️ Project Structure

```
logistics-analysis/
├── analysis.py                          # Main Python script (simulation + EDA + charts)
├── requirements.txt                     # Python dependencies
├── logistics_data.csv                   # Generated dataset (2,000 rows)
├── summary_statistics.csv               # Descriptive stats table
├── correlation_matrix.csv               # Pearson correlation matrix
├── by_mode.csv                          # Aggregated metrics by transport mode
├── by_carrier.csv                       # Aggregated metrics by carrier
├── by_region.csv                        # Aggregated metrics by region
├── monthly.csv                          # Monthly trend data
├── charts/                              # 10 generated PNG charts (200 dpi)
│   ├── 01_delivery_time_distribution.png
│   ├── 02_cost_by_mode_boxplot.png
│   ├── 03_distance_vs_cost_scatter.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_monthly_trend.png
│   ├── 06_ontime_by_carrier.png
│   ├── 07_region_mode_heatmap.png
│   ├── 08_delivery_time_by_region_violin.png
│   ├── 09_pairplot_key_variables.png
│   └── 10_mode_mix_by_region_stacked.png
├── Logistics_Analysis_Report.docx       # Final written report with embedded charts
└── README.md
```

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<username>/logistics-data-analysis.git
cd logistics-data-analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analysis
python analysis.py
```

The script prints EDA results to the terminal and saves all charts in the `charts/` folder.

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **pandas** | Data manipulation and aggregation |
| **NumPy** | Numerical simulation and computation |
| **Matplotlib** | Base charting library |
| **Seaborn** | Statistical visualizations |
| **SciPy** | Kruskal-Wallis hypothesis test |

## 📈 Visualizations Preview

| # | Chart | Purpose |
|---|-------|---------|
| 1 | Delivery Time Histogram | Distribution shape, centre, and spread |
| 2 | Cost Box Plot by Mode | Median comparison, spread, outliers |
| 3 | Distance vs Cost Scatter | Pairwise relationship, mode clusters |
| 4 | Correlation Heatmap | Multi-variable correlation overview |
| 5 | Monthly Trend (dual-axis) | Seasonal patterns in time and reliability |
| 6 | On-Time Rate by Carrier | Carrier performance comparison |
| 7 | Region × Mode Heatmap | Cross-tabulation of delivery time |
| 8 | Violin Plot by Region | Full distribution shape per region |
| 9 | Pair Plot | All pairwise relationships at once |
| 10 | Stacked Bar — Mode Mix | Transport mode share per region |

## 📝 Report

The detailed written report (`Logistics_Analysis_Report.docx`) includes:
- Methodology overview
- Python code excerpts with explanations
- All 10 visualizations with interpretations
- Analytical conclusions and business recommendations

## ⚠️ Note

The dataset is **simulated** using a fixed random seed (42) for full reproducibility.  Results demonstrate analytical techniques but do not represent a real company's data.

## 📄 License

This project is for educational and internship evaluation purposes.

---

*Built with Python, pandas, NumPy, Matplotlib, Seaborn, and SciPy.*
