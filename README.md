# 🛒 Customer Segmentation — Unsupervised Machine Learning

> *Identifying hidden customer groups in wholesale distributor data using PCA + K-Means Clustering*

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

A wholesale distributor serving 440 clients across Lisbon, Portugal needed to understand **who their customers actually are** — without any pre-existing labels. This project applies **unsupervised machine learning** to discover natural customer groupings based on annual spending across six product categories.

### Business Problem
The distributor wanted to:
- Optimize delivery logistics per customer type
- Tailor marketing strategies to each segment
- Forecast demand more accurately by customer group
- Allocate sales resources to high-value accounts

### Solution
Applied an end-to-end ML pipeline: **EDA → Log Transform → Outlier Detection → PCA → K-Means Clustering → Business Interpretation**

### Result
Discovered **2 well-separated customer segments** that align with the real distribution channel labels (HoReCa vs. Retail) — validating the model with a **Silhouette Score of 0.43**.

---

## 🗂️ Project Structure

```
customer-segmentation/
│
├── 📓 customer_segments.ipynb    # Main analysis notebook (complete)
├── 📊 customers.csv              # Wholesale Customers Dataset (UCI ML Repo)
├── 🐍 visuals.py                 # Helper visualization functions
├── 📁 plots/                     # Generated visualization outputs
│   ├── 01_distributions_raw.png
│   ├── 02_boxplots.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_scatter_matrix.png
│   ├── 05_distributions_log.png
│   ├── 06_pca_components.png
│   ├── 07_explained_variance.png
│   ├── 08_biplot.png
│   ├── 09_elbow_silhouette.png
│   ├── 10_cluster_visualization.png
│   ├── 11_channel_validation.png
│   └── 12_cluster_profiles.png
└── 📄 README.md
```

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Wholesale Customers](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)

| Feature | Description | Type |
|---|---|---|
| `Fresh` | Annual spending on fresh products | Continuous (m.u.) |
| `Milk` | Annual spending on dairy products | Continuous (m.u.) |
| `Grocery` | Annual spending on grocery items | Continuous (m.u.) |
| `Frozen` | Annual spending on frozen foods | Continuous (m.u.) |
| `Detergents_Paper` | Annual spending on detergents & paper | Continuous (m.u.) |
| `Delicatessen` | Annual spending on deli/specialty foods | Continuous (m.u.) |

**440 customers · 6 spending features · No missing values**

---

## 🔬 Methodology

```
Raw Data (440×8)
    ↓
Drop Channel & Region
    ↓
EDA: Distributions, Correlations, Skewness
    ↓
Log Transformation (normalize skewed financial data)
    ↓
Outlier Removal via Tukey's IQR Method
    ↓ (435 clean records)
PCA — Full (6D) → Variance Analysis
    ↓
PCA — Reduced (2D) for clustering
    ↓
K-Means (k=2) + Elbow + Silhouette
    ↓
Validate vs. Known Channel Labels
    ↓
Business Interpretation & Recommendations
```

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| **Optimal Clusters** | 2 |
| **Silhouette Score** | 0.4263 |
| **PCA Variance Retained** | 70.7% (2D) |
| **PC1 Variance** | 44.3% |
| **Cluster 0 (Food-Service)** | 258 customers (59%) |
| **Cluster 1 (Retail)** | 177 customers (41%) |

### Segments Discovered

**🍽️ Cluster 0 — Food-Service** (Restaurants, Hotels, Cafés)
- High Fresh spending (~8,878 m.u.)
- Low Grocery & Detergents_Paper
- Perishable-focused, daily replenishment needs

**🏪 Cluster 1 — Retail** (Supermarkets, Convenience Stores)
- High Grocery (~12,050 m.u.) & Detergents_Paper (~4,526 m.u.)
- Low Fresh spending
- Packaged goods, predictable bulk orders

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data manipulation & EDA |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | PCA, K-Means, GMM, Silhouette |
| `Jupyter Notebook` | Interactive analysis environment |

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation.git
cd customer-segmentation
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Launch the Notebook
```bash
jupyter notebook customer_segments.ipynb
```

> **Note:** Run all cells in order (Kernel → Restart & Run All) for reproducible results.

---

## 📈 Visual Outputs

| Plot | Description |
|---|---|
| Distribution plots | Raw vs. log-transformed feature distributions |
| Correlation heatmap | Feature relationship matrix |
| PCA component chart | Feature weights per principal component |
| Explained variance | Cumulative variance by PC |
| Biplot | Original features projected onto PCA space |
| Elbow + Silhouette | K selection methodology |
| Cluster scatter | K-Means results in 2D PCA space |
| Channel validation | Comparing clusters to ground truth labels |
| Cluster profiles | Spending by segment in original scale |

---

## 🔮 Future Improvements

- [ ] Hierarchical clustering (dendrogram) to explore sub-segments
- [ ] DBSCAN for irregular cluster detection
- [ ] Incorporate `Region` for geographic demand analysis
- [ ] Build an API endpoint to score new customers at onboarding
- [ ] Time-series extension if periodic spending data becomes available
- [ ] Dimensionality reduction comparison: t-SNE, UMAP vs. PCA

---

## 📚 References

- [UCI ML Repository — Wholesale Customers Dataset](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)
- Marques, A. & Tilley, D. (2011). Wholesale Customer Dataset. UCI Machine Learning Repository.
- Jain, A.K. (2010). Data clustering: 50 years beyond K-means. *Pattern Recognition Letters*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*.

---

## 👤 Author

**[Your Name]**
- 🌐 Portfolio: [your-portfolio.com](https://your-portfolio.com)
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*If this project helped you, please ⭐ the repo!*
