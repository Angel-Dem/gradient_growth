# Pyaterochka Store Revenue Forecasting

A team solution developed for the **Gradient of Growth online hackathon**, organized by **Sber and the HSE Faculty of Computer Science**, based on a business case provided by **X5 Group**.

> **Team result:** advanced to the second stage, placed **15th**, and achieved a score of **90 out of 100**.

## Project Overview

The objective of the project was to build a machine learning model capable of forecasting the **monthly retail turnover (revenue)** of Pyaterochka stores.

The available data included:

- monthly store revenue history;
- store characteristics;
- geographic information;
- population and household statistics;
- pedestrian and vehicle traffic;
- nearby infrastructure and competitors;
- number of checkout counters;
- alcohol license availability.

The problem was approached as a collection of short time series. For each store, the model had to capture its individual revenue level, temporal dynamics, and static characteristics.

## Solution Pipeline

1. Exploratory analysis of store-level time series.
2. Analysis of revenue trends, seasonality, and distribution.
3. Autocorrelation and feature correlation analysis.
4. Creation of lag-based, rolling, and trend features.
5. Walk-forward validation without shuffling temporal observations.
6. Training a Ridge regression baseline.
7. Training a Gradient Boosting model.
8. Model comparison and feature importance analysis.
9. Generation of the final forecast and `submission_final.csv`.

## Dataset

Each row in the training dataset represents one store during a particular month.

### Target Variable

- `РТО` — monthly retail turnover of a store.

### Temporal Information

- month number;
- previous revenue values;
- rolling statistics;
- revenue growth rates;
- store-specific trend;
- time-series volatility.

### Static Store Features

- store opening period;
- retail floor area;
- city and region;
- population;
- number of households;
- pedestrian and vehicle traffic;
- nearby infrastructure within predefined radii;
- nearby grocery stores and other Pyaterochka locations;
- number of checkout counters;
- alcohol license availability.

The original hackathon dataset is not included in this repository.

## Exploratory Data Analysis

### Revenue Dynamics and Distribution

The analysis showed substantial differences in absolute revenue levels across stores. However, most individual time series were relatively stable, making previous revenue values especially useful predictors.

![Time-series analysis](fig1_trends.png)

### Time-Series Decomposition

The average monthly revenue series was decomposed into:

- trend;
- seasonal component;
- residual noise.

Because the available history covered only ten months, a full annual seasonal decomposition was not possible. Instead, the trend was estimated using a linear model, while short-term deviations were analysed separately.

![Time-series decomposition](fig2_decomposition.png)

### Statistical Analysis

The following properties were investigated:

- autocorrelation;
- first differences;
- store-specific trends;
- coefficient of variation;
- correlations between static store characteristics and revenue.

Lag-based features and variables representing store scale, especially the number of checkout counters, were among the most informative predictors.

![Statistical analysis](fig3_stats.png)

## Feature Engineering

All features for a target month were constructed exclusively from information available before that month. This prevents data leakage from future observations.

### Lag Features

- `lag1` — revenue in the previous month;
- `lag2` — revenue two months earlier;
- `lag3` — revenue three months earlier.

### Rolling Statistics

- `ma3` — average revenue over the previous three months;
- `ma6` — average revenue over the previous six months;
- `ma_all` — average over the entire available history;
- `std_all` — standard deviation;
- `cv` — coefficient of variation.

### Dynamic Features

- `diff1` — absolute change in revenue from the previous month;
- `growth` — relative revenue growth rate;
- `ratio_last` — ratio of the latest value to the historical mean;
- `trend` — slope of the linear revenue trend;
- `trend_norm` — normalized trend slope;
- historical median and range.

Categorical store characteristics were converted into numerical values using `LabelEncoder`.

## Validation Strategy

A random train-test split was not used because the observations are time-dependent.

Instead, the project used **walk-forward validation**:

- months 4–9 were predicted sequentially to construct the training set;
- month 10 was used as a held-out temporal validation period;
- after model evaluation, the final forecast was generated for month 11.

This approach evaluates the model under conditions that closely resemble real-world forecasting, where only past information is available.

## Models

### Ridge Regression

Ridge regression was used as an interpretable linear baseline.

Before training, numerical features were standardized with `StandardScaler`. L2 regularization reduced the impact of multicollinearity between lag features and rolling averages.

### Gradient Boosting

The primary nonlinear model was `GradientBoostingRegressor`.

The model can capture:

- nonlinear relationships;
- interactions between temporal and static features;
- differences across regions and store types;
- heterogeneous store-level revenue dynamics.

The final prediction was produced using an ensemble of Ridge regression and Gradient Boosting:

```python
final_prediction = 0.3 * ridge_prediction + 0.7 * gradient_boosting_prediction
```

## Evaluation Metric

The primary evaluation metric was **Mean Absolute Percentage Error (MAPE)**:

```text
MAPE = mean(|y_true - y_pred| / |y_true|) × 100%
```

A lower MAPE indicates a more accurate forecast. In the competition scoring logic, model performance was represented using a score related to `100 − MAPE`.

## Competition Result

Our team:

- advanced to the second stage of the hackathon;
- placed **15th**;
- achieved a score of **90 out of 100**.

The project provided practical experience in time-series analysis, feature engineering, leakage-safe temporal validation, gradient boosting, and model ensembling.

## Repository Structure

```text
.
├── analysis_notebook.py          # EDA, feature engineering, and model training
├── fig1_trends.png               # revenue dynamics and distribution analysis
├── fig2_decomposition.png        # decomposition of the average time series
├── fig3_stats.png                # statistical analysis
├── train.csv                     # original data, not included
├── submission_final.csv          # generated final forecast
└── README.md
```

## Running the Project

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

### 4. Add the Dataset

Place `train.csv` in the root directory of the project.

### 5. Run the Analysis

```bash
python analysis_notebook.py
```

The script will generate the visualizations, train the models, and create:

```text
submission_final.csv
```

## Technology Stack

- Python;
- pandas;
- NumPy;
- Matplotlib;
- SciPy;
- scikit-learn;
- Ridge Regression;
- Gradient Boosting;
- time-series feature engineering;
- walk-forward validation.

## Limitations

- The dataset contains only ten months of history, making reliable annual seasonality estimation impossible.
- Static store characteristics do not change over time.
- Competition data is not included in the public repository.
- Model performance may be affected by distribution shifts in future revenue periods.

## Potential Improvements

- use CatBoost or LightGBM with native categorical feature handling;
- train separate models for different store groups;
- apply target encoding to regions and cities;
- tune hyperparameters using time-series cross-validation;
- model the logarithm of revenue;
- optimize the model directly for MAPE;
- use a longer historical period to capture annual seasonality;
- analyse forecast errors by region, store size, and revenue level.
