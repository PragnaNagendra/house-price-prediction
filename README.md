# House Price Prediction

An end-to-end machine-learning project for predicting residential sale prices using feature engineering, cross-validation and ensemble regression models.

## Project Overview

This project uses the Kaggle **House Prices: Advanced Regression Techniques** dataset. The objective is to predict the final sale price of each property from its numerical and categorical characteristics.

The workflow covers:

* Missing-value treatment
* Categorical encoding
* Feature engineering
* Skewness correction and scaling
* Model training and validation
* Ensemble prediction
* Kaggle submission-file generation

## Dataset

The project uses the Ames Housing data supplied through the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

The dataset contains information about property size, quality, location, condition, rooms, garages and other characteristics that may influence sale price.

## Methodology

### Data Preprocessing

* Imputed missing numerical and categorical values.
* Applied appropriate transformations to skewed numerical features.
* Encoded categorical variables for model training.
* Standardised selected features where required.

### Feature Engineering

Created additional variables representing information such as:

* Total living and basement area
* Total number of bathrooms
* Property and garage age
* Overall property quality and condition

### Models Evaluated

* Linear Regression
* LASSO Regression
* ElasticNet
* Gradient Boosting Regressor
* XGBoost
* LightGBM
* Bagging Regressor
* Stacked and blended predictions

## Results

Model performance was evaluated using Root Mean Squared Error (RMSE). Lower values indicate better predictive performance.

| Model                       |  RMSE |
| --------------------------- | ----: |
| ElasticNet                  | 0.115 |
| Gradient Boosting Regressor | 0.118 |
| XGBoost Regressor           | 0.119 |
| LightGBM Regressor          | 0.123 |
| LASSO Regression            | 0.125 |
| Linear Regression           | 0.160 |
| Bagging Regressor           | 0.172 |

ElasticNet produced the lowest individual-model RMSE in the recorded validation results. Predictions from multiple strong models were also combined to improve robustness.

## Repository Structure

```text
house-price-prediction/
├── house_price_prediction.ipynb
└── README.md
```

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/PragnaNagendra/house-price-prediction.git
cd house-price-prediction
```

2. Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm sklearn-pandas jupyter
```

3. Download `train.csv` and `test.csv` from Kaggle and place them in the project directory.

4. Start Jupyter Notebook:

```bash
jupyter notebook house_price_prediction.ipynb
```

Update the dataset paths inside the notebook if your files are stored elsewhere.

## Tools and Technologies

* Python
* pandas and NumPy
* Matplotlib and Seaborn
* scikit-learn
* XGBoost
* LightGBM
* Jupyter Notebook

## Limitations and Future Improvements

* Validate the models using repeated cross-validation.
* Add automated hyperparameter optimisation.
* Include SHAP-based model interpretation.
* Refactor preprocessing and modelling into reusable Python pipelines.
* Add automated testing and dependency management.

## Author

**Nagendra Pragna Marella**

MSc Data Science candidate at Queen Mary University of London.

[LinkedIn](https://www.linkedin.com/in/nagendra-pragna-marella/) | [GitHub](https://github.com/PragnaNagendra)
