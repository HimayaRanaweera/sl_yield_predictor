# Sri Lanka Yield Predictor
A data mining project that predicts agricultural crop yields in Sri Lanka using machine learning models, which is developed by Group Mining Minds.

## Project Overview
This project applies data mining techniques to predict crop yield (mt per hectare) in Sri Lanka using agricultural and climatic data. The solution helps farmers, policymakers and planners make informed decisions about production and resource management.

## Motivation & Objectives
- Agriculture is a critical sector in Sri Lanka, but yield outcomes are uncertain.  
- Factors like rainfall, temperature, fertilizer and soil type make prediction difficult.  
- Our objective is to:
  1. Clean and preprocess agricultural datasets.
  2. Train and compare multiple machine learning models.
  3. Build an interactive application for yield prediction.

## Scope of Work
- Data collection & cleaning (~20,000 records).  
- Exploratory Data Analysis (EDA).  
- Feature engineering.  
- Training models: Linear Regression, Random Forest, Gradient Boosting, XGBoost, etc.  
- Model evaluation using RMSE, MAE, R².  
- Interactive Hugging Face Spaces web application for predictions.

## Dataset & Features
- Features: Year, Season, Province, Crop, Soil Type, Irrigation, Rainfall, Temperature, Fertilizer, Market Price, etc.  
- Target variable: Yield per hectare (mt/ha).  
- Handling missing values:
  - Median for numerical features.
  - Mode / "missing" token for categorical features.
  - Rows with missing target values removed.

## Methodology
1. **Data Preprocessing** – missing value handling, encoding categorical variables, scaling.  
2. **Feature Engineering** – transformations (log), derived features.  
3. **Model Training** – compared 8 algorithms including Gradient Boosting and XGBoost.  
4. **Evaluation** – metrics: RMSE, MAE, R².  
5. **Best Model** – Tuned Gradient Boosting achieved best performance.

## Results
- Gradient Boosting achieved lowest RMSE and high R².  
- Outperformed baseline (Dummy Regressor).  
- Strong predictors: rainfall, fertilizer usage, crop type.  

## Application (Hugging Face Spaces App)
We deployed the final model using Hugging Face Spaces (Gradio).
The app allows users to:
- Input variables such as rainfall, fertilizer, soil type, crop, etc.
- Get predicted yield per hectare and estimated production.
- View model performance details.

## Usage Instructions (Local)
If you want to run locally:
Clone this repository:
git clone https://github.com/HimayaRanaweera/sl_yield_predictor.git
cd sl_yield_predictor

Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # (or venv\Scripts\activate on Windows)

Install dependencies:
pip install -r requirements.txt

Run the Hugging Face Gradio app:
python app.py

## Challenges & Solutions
- **Missing Data** – solved with imputation (median/mode).  
- **Data Leakage** – avoided by selecting only pre-harvest features.  
- **Model Overfitting** – handled via cross-validation and hyperparameter tuning.

## Future Work
- Add external data (pests, extreme weather, policy data).  
- Provide visualization dashboards.  
- Mobile application and multilingual support 




