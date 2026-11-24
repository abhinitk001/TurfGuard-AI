# TurfGuard AI
### An Explainable Framework for NFL Player Biomechanics, Playing Surface Analysis and Injury Pattern Investigation

TurfGuard AI is a data analytics and machine learning project that investigates the relationship between NFL player movement, playing surfaces, environmental conditions, and recorded lower-limb injuries.

Using the **NFL 1st and Future – Analytics** dataset from Kaggle, the project explores player movement patterns, compares natural and synthetic turf, studies injury-related patterns, and develops an experimental machine learning model to analyze recorded injury labels.

The project also includes an interactive Streamlit dashboard for exploring the dataset, surface comparisons, and model outputs.

---

## Project Objectives

- Analyze NFL player movement using player tracking data.
- Compare movement characteristics on natural and synthetic playing surfaces.
- Investigate recorded lower-limb injuries and their distribution.
- Explore the relationship between player movement, environmental conditions, and injury records.
- Develop an experimental machine learning model for recorded injury-label analysis.
- Present analytical findings through an interactive dashboard.

## Dataset

**Source:** [NFL 1st and Future – Analytics (Kaggle)](https://www.kaggle.com/competitions/nfl-playing-surface-analytics)

The project uses three primary datasets:

| Dataset | Description |
|---|---|
| `InjuryRecord.csv` | Contains recorded player injury information. |
| `PlayList.csv` | Contains game, play, player, position, and surface metadata. |
| `PlayerTrackData.csv` | Contains detailed player tracking information, including position, speed, direction, and movement. |

The tracking data is used to derive movement-related features such as distance, speed, acceleration estimates, and movement ranges.

**Note:** The original Kaggle datasets are not included in this repository because of their size. Download them from Kaggle and place them in the appropriate data directory.

## Project Workflow

The project is organized into the following stages:

| Notebook | Description |
|---|---|
| `01_data_understanding.ipynb` | Initial exploration and understanding of the datasets. |
| `02_data_cleaning.ipynb` | Data cleaning and preparation. |
| `03_data_integration.ipynb` | Integration of injury records and play metadata. |
| `04_feature_engineering.ipynb` | Creation of movement and player-related features. |
| `05_surface_analysis.ipynb` | Comparison of natural and synthetic playing surfaces. |
| `06_model_development.ipynb` | Development of machine learning models. |
| `07_model_evaluation.ipynb` | Model evaluation and performance analysis. |
| `08_model_testing.ipynb` | Experiments and testing on selected and held-out data. |

## Key Features

### 1. Player Movement Analysis
- Average and maximum speed.
- Total movement distance.
- Speed variation.
- Estimated acceleration.
- Movement duration and spatial ranges.

### 2. Playing Surface Analysis
- Natural vs. synthetic turf comparison.
- Statistical comparison of movement features.
- Injury record distribution across surfaces.
- Body-part injury analysis.

### 3. Machine Learning
The project explores machine learning approaches, including:

- Logistic Regression
- CatBoost

The models analyze recorded injury labels using engineered player-play features.

### 4. Model Evaluation
Evaluation includes:

- Confusion matrix
- ROC-AUC
- Precision and recall
- PR-AUC
- Threshold experiments
- Top-ranked record analysis

### 5. Interactive Dashboard
The Streamlit dashboard provides:

- Project overview and dataset statistics.
- AI analysis and model scoring.
- Playing surface comparisons.
- High-score record exploration.
- Model evaluation visualization.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, CatBoost |
| Model Storage | Joblib |
| Dashboard | Streamlit |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git and GitHub |

## Repository Structure

```text
TurfGuard-AI/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_data_integration.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_surface_analysis.ipynb
│   ├── 06_model_development.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_model_testing.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Large datasets and trained model files are excluded from version control.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/TurfGuard-AI.git
```

### 2. Navigate to the Project Directory

```bash
cd TurfGuard-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the Dataset

Download the competition dataset from Kaggle and organize it according to the notebook paths.

Example:

```text
data/
├── raw/
│   ├── InjuryRecord.csv
│   ├── PlayList.csv
│   └── PlayerTrackData.csv
│
└── processed/
```

### 6. Run the Jupyter Notebooks

```bash
jupyter notebook
```

Run the notebooks in sequence to reproduce the analysis and model development workflow.

### 7. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Make sure the required processed datasets and compatible trained model are available in their expected locations.

---

## Important Research Limitations

- The project analyzes recorded injury labels, not medically confirmed injury risk.
- Some injury records do not contain a matching `PlayKey`, limiting record-level matching.
- The dataset is highly imbalanced, with very few matched positive injury records.
- A play without a matched injury record should not automatically be interpreted as an injury-free play.
- Acceleration is estimated from changes in recorded speed over time.
- Model results are exploratory and should not be interpreted as a reliable medical prediction system.
- The analysis identifies patterns and associations; it does not establish causation.

## Project Goal

The goal of TurfGuard AI is to explore how playing surfaces, environmental conditions, and player movement characteristics relate to recorded NFL injury patterns.

Rather than relying only on predictive accuracy, the project emphasizes data exploration, statistical analysis, explainability, and meaningful interpretation of player movement.

## Acknowledgements

- Kaggle – NFL 1st and Future: Analytics
- National Football League (NFL) dataset contributors
- Open-source Python and machine learning community

---

**TurfGuard AI | Data Analytics • Machine Learning • Sports Injury Research**