#Google Collab Link 
#https://colab.research.google.com/drive/1LKMNIuqm_EJZ2hGquf_g2lvV5EUht5sz?usp=sharing
Machine Learning project: Random Forest weather forecasting model with month-based train/val/test split.
🌦️ Weather Forecasting Model (Random Forest)

A machine learning model that predicts future temperature values using historical weather data.
This project demonstrates a full ML workflow: data cleaning, feature engineering, month-based time-series splitting, hyperparameter tuning, evaluation, and model saving.

📌 Project Overview

The goal of this project is to forecast temperature using a Random Forest Regressor trained on past climate variables.
This is also a beginner-friendly introduction to time-series forecasting, train/val/test splits, and hyperparameter optimization.

🧠 Key Features

Cleaned and preprocessed weather dataset

Feature engineering using lag values, rolling averages, and month encoding

Strict time-aware splitting:

Train: Jan–Aug

Validation: Sep–Oct

Test: Nov–Dec

Hyperparameter tuning using GridSearchCV + PredefinedSplit

Saved trained model (best_model.joblib)

Forecast vs Actual visualizations

Evaluation metrics (MAE, MSE, RMSE, R²)

🗂️ Project Structure
weather-forecasting-model/
│
├── notebook.ipynb                  # Final Jupyter Notebook
├── best_model.joblib               # Saved Random Forest model
├── actual_vs_predicted.png         # Forecast vs Actual plot
├── residuals.png                   # Residuals distribution
├── sample_predictions.csv          # Model outputs (optional)
├── requirements.txt                # Dependencies
└── README.md                       # This file

🧼 Data Preprocessing Steps

Handled missing values

Converted date/time columns

Added lag features (t-1, t-2, t-3…)

Added rolling averages (7-day, 30-day)

Extracted month numbers

Removed future leakage

Normalized/standardized features (if required)

🔀 Train–Validation–Test Split (VERY IMPORTANT)

To avoid data leakage:

Split	Months Used
Train	Jan → Aug (1–8)
Validation	Sept → Oct (9–10)
Test	Nov → Dec (11–12)

Validation set was used for hyperparameter tuning using PredefinedSplit.

🔧 Modeling & Hyperparameter Tuning

Model used: RandomForestRegressor

Tuning was done with:

GridSearchCV

PredefinedSplit (custom month-based val set)

Scoring metric: neg_mean_squared_error

Best parameters found:

n_estimators: 200/400 (Example)
max_depth: 10/None
min_samples_split: 2
min_samples_leaf: 1


(Your actual values may differ — replace them if needed.)

📊 Results
Metrics (Test Set: Nov–Dec)

MAE: X.XX

MSE: X.XX

RMSE: X.XX

R² Score: X.XX

(Fill in with your actual numbers.)

📈 Visualizations
Forecast vs Actual

Residual Distribution

💾 Using the Saved Model
import joblib
model = joblib.load("best_model.joblib")

# Predict new data
preds = model.predict(new_df)

🛠️ How to Run
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/weather-forecasting-model.git

2. Install dependencies
pip install -r requirements.txt

3. Open notebook
jupyter notebook notebook.ipynb

🚀 Future Improvements

Add humidity/wind/rain features

Try Gradient Boosting or XGBoost

Build a Streamlit web app

Deploy using HuggingFace Spaces

Add cross-validation backtesting

📜 License

MIT License — free to use & modify.
