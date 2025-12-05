# Heart Failure Prediction: Machine Learning for Clinical Risk Assessment 🏥

A comprehensive machine learning project that predicts heart failure mortality risk using clinical data. This project includes exploratory data analysis, multiple ML models, SHAP-based interpretability, and an interactive Streamlit dashboard for real-time risk prediction.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Model Performance](#model-performance)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results & Insights](#results--insights)
- [Contributing](#contributing)

## 🎯 Project Overview

Heart failure is a serious medical condition affecting millions worldwide. This project leverages machine learning to predict patient mortality risk based on clinical features, providing:

- **Predictive Models**: Four state-of-the-art ML algorithms trained and compared
- **Model Interpretability**: SHAP (SHapley Additive exPlanations) for transparent predictions
- **Interactive Dashboard**: User-friendly Streamlit application for risk assessment
- **Clinical Insights**: Actionable recommendations based on risk stratification

### 🎓 Use Cases
- Clinical decision support for healthcare professionals
- Research and educational demonstrations
- Risk stratification for patient management
- Model performance evaluation and comparison

## 📊 Dataset Description

**Source**: Heart Failure Clinical Records Dataset  
**Patients**: 299 individuals  
**Features**: 12 clinical and demographic variables  
**Target**: Death event (binary classification)

### Clinical Features

| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| **age** | Patient age | Numeric | 40-95 years |
| **anaemia** | Decrease of red blood cells or hemoglobin | Binary | 0 = No, 1 = Yes |
| **creatinine_phosphokinase** | Level of CPK enzyme in blood | Numeric | 23-7861 mcg/L |
| **diabetes** | Presence of diabetes | Binary | 0 = No, 1 = Yes |
| **ejection_fraction** | Percentage of blood leaving heart | Numeric | 14-80% |
| **high_blood_pressure** | Presence of hypertension | Binary | 0 = No, 1 = Yes |
| **platelets** | Platelet count in blood | Numeric | 25,100-850,000 kiloplatelets/mL |
| **serum_creatinine** | Level of creatinine in blood | Numeric | 0.5-9.4 mg/dL |
| **serum_sodium** | Level of sodium in blood | Numeric | 113-148 mEq/L |
| **sex** | Gender | Binary | 0 = Female, 1 = Male |
| **smoking** | Smoking status | Binary | 0 = No, 1 = Yes |
| **time** | Follow-up period | Numeric | 4-285 days |

### Dataset Statistics
- **Total Samples**: 299
- **Class Distribution**: 
  - Survived: 203 patients (67.9%)
  - Death Event: 96 patients (32.1%)
- **Class Imbalance**: Addressed using SMOTE (Synthetic Minority Over-sampling Technique)
- **Missing Values**: None
- **Duplicate Records**: None

## 🏆 Model Performance

Four machine learning models were trained and evaluated using stratified cross-validation:

### ROC-AUC Scores (Primary Metric)

| Rank | Model | Train ROC-AUC | Test ROC-AUC | Status |
|------|-------|---------------|--------------|--------|
| 🥇 | **Random Forest** | 0.9882 | **0.9089** | ⭐ Best Model |
| 🥈 | **LightGBM** | 1.0000 | 0.8665 | Strong |
| 🥉 | **XGBoost** | 1.0000 | 0.8485 | Strong |
| 4th | **Logistic Regression** | 0.9052 | 0.8588 | Baseline |

### Best Model: Random Forest 🌲

**Test Set Performance:**
- **ROC-AUC**: 0.9089 (Excellent discrimination)
- **Accuracy**: 85.0%
- **Precision**: 81.25%
- **Recall**: 68.42%
- **F1-Score**: 74.29%

**Why Random Forest?**
- Best generalization (highest test ROC-AUC)
- No overfitting (unlike XGBoost and LightGBM with perfect train scores)
- Robust performance across all metrics
- Excellent feature importance interpretability

### Comprehensive Metrics Comparison

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
|-------|---------------|----------------|-------------|---------|
| Random Forest | **85.00%** | 81.25% | **68.42%** | **74.29%** |
| XGBoost | **85.00%** | **85.71%** | 63.16% | 72.73% |
| LightGBM | 81.67% | 78.57% | 57.89% | 66.67% |
| Logistic Regression | 81.67% | 78.57% | 57.89% | 66.67% |

## ✨ Key Features

### 1. 📓 Jupyter Notebooks (Complete ML Pipeline)
- **01_eda_and_cleaning.ipynb**: Data exploration and preprocessing
- **02_model_training.ipynb**: Model development and hyperparameter tuning
- **03_model_interpretation.ipynb**: SHAP analysis and feature importance
- **04_model_evaluation.ipynb**: Comprehensive performance evaluation

### 2. 🚀 Interactive Dashboard
- **Real-time Predictions**: Input patient data and get instant risk assessments
- **SHAP Explanations**: Understand which features drive each prediction
- **Visual Analytics**: ROC curves, confusion matrices, correlation heatmaps
- **Risk Stratification**: Color-coded risk categories (Low/Medium/High)
- **Clinical Recommendations**: Actionable insights based on risk level

### 3. 🔍 Model Interpretability
- SHAP waterfall plots for individual predictions
- Feature importance rankings
- Top 3 contributing factors with human-readable explanations
- Confidence scores for predictions

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd heart-failure-project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- Data science: pandas, numpy, scipy
- Visualization: matplotlib, seaborn
- Machine learning: scikit-learn, xgboost, lightgbm, imbalanced-learn
- Interpretability: shap
- Dashboard: streamlit
- Utilities: joblib, openpyxl

## 📖 Usage

### Option 1: Run the Interactive Dashboard (Recommended)

```bash
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`

**Dashboard Pages:**
1. **🏠 Home**: Project overview and key findings
2. **📊 Data Overview**: Dataset statistics and visualizations
3. **🎯 Model Performance**: ROC curves and metrics comparison
4. **🔮 Risk Prediction Tool**: Interactive patient risk assessment

### Option 2: Run Jupyter Notebooks

```bash
jupyter notebook
```

Execute the notebooks in order:
1. `01_eda_and_cleaning.ipynb` - Explore the data
2. `02_model_training.ipynb` - Train models
3. `03_model_interpretation.ipynb` - Analyze with SHAP
4. `04_model_evaluation.ipynb` - Evaluate performance

### Option 3: Use Models Programmatically

```python
import joblib
import pandas as pd

# Load the best model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare patient data
patient_data = pd.DataFrame({
    'age': [65],
    'anaemia': [0],
    'creatinine_phosphokinase': [582],
    'diabetes': [0],
    'ejection_fraction': [20],
    'high_blood_pressure': [0],
    'platelets': [265000],
    'serum_creatinine': [1.9],
    'serum_sodium': [130],
    'sex': [1],
    'smoking': [0],
    'time': [4]
})

# Scale and predict
patient_scaled = scaler.transform(patient_data)
risk_probability = model.predict_proba(patient_scaled)[0][1]

print(f"Mortality Risk: {risk_probability*100:.1f}%")
```

## 📁 Project Structure

```
heart-failure-project/
│
├── 01_eda_and_cleaning.ipynb          # Data exploration
├── 02_model_training.ipynb            # Model development
├── 03_model_interpretation.ipynb      # SHAP analysis
├── 04_model_evaluation.ipynb          # Performance evaluation
├── app.py                             # Streamlit dashboard
│
├── heart_failure_clinical_records_dataset.csv  # Raw data
│
├── models/                            # Saved models and artifacts
│   ├── random_forest_model.pkl        # Best model (Random Forest)
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl                     # StandardScaler
│   ├── shap_explainer.pkl             # SHAP explainer
│   ├── model_comparison.csv           # Metrics comparison
│   └── *.png                          # Visualizations
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── README_DASHBOARD.md                # Dashboard documentation
├── COMPLETION_REPORT.md               # Project status report
└── PREDICTION_TOOL_GUIDE.md           # Detailed tool guide
```

## 🔧 Technical Details

### Machine Learning Pipeline
1. **Data Preprocessing**
   - Feature scaling using StandardScaler
   - No missing value imputation needed (clean dataset)
   - Train-test split: 80-20 with stratification

2. **Class Imbalance Handling**
   - SMOTE applied to training set only
   - Original class distribution: 68% survived, 32% death
   - Balanced training improves minority class recall

3. **Model Training**
   - Stratified K-Fold Cross-Validation (5 folds)
   - Hyperparameter tuning via GridSearchCV
   - Models: Logistic Regression, Random Forest, XGBoost, LightGBM

4. **Model Evaluation**
   - Primary metric: ROC-AUC (handles class imbalance)
   - Secondary metrics: Accuracy, Precision, Recall, F1-Score
   - Confusion matrix analysis
   - ROC curve comparison

5. **Model Interpretation**
   - SHAP (TreeExplainer for tree-based models)
   - Global feature importance
   - Local explanations for individual predictions
   - Feature interaction analysis

### Key Technologies
- **Python**: 3.8+
- **scikit-learn**: ML models and preprocessing
- **XGBoost & LightGBM**: Gradient boosting algorithms
- **SHAP**: Model interpretability
- **Streamlit**: Interactive dashboard
- **pandas & numpy**: Data manipulation
- **matplotlib & seaborn**: Visualizations

## 📈 Results & Insights

### Top Predictive Features (SHAP Analysis)

1. **⏱️ Time (Follow-up Period)**: Most important predictor
   - Longer follow-up correlates with survival
   - Short follow-up periods indicate higher risk

2. **💪 Ejection Fraction**: Critical cardiac function metric
   - Low ejection fraction (<30%) significantly increases risk
   - Every 10% decrease raises mortality probability

3. **🩸 Serum Creatinine**: Kidney function indicator
   - Elevated levels (>1.5 mg/dL) associated with poor prognosis
   - Reflects overall organ function deterioration

4. **🧂 Serum Sodium**: Electrolyte balance
   - Hyponatremia (<135 mEq/L) indicates severity
   - Critical for fluid management

5. **👤 Age**: Demographic risk factor
   - Age >65 increases baseline risk
   - Combined effect with other factors

### Clinical Insights

- **High-Risk Profile**: Age >70, EF <25%, serum creatinine >2.0, short follow-up
- **Protective Factors**: Normal EF (>40%), normal kidney function, longer survival time
- **Intervention Priorities**: Monitor kidney function, optimize cardiac output, manage fluids

### Model Limitations
- Retrospective dataset (not prospective validation)
- Limited sample size (299 patients)
- Single-center data (generalizability)
- Temporal aspects (follow-up time as predictor)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- External validation on different datasets
- Deep learning models (neural networks)
- Additional clinical features
- Longitudinal analysis
- Deployment to cloud platforms

## 📄 License

This project is for educational and research purposes.

## 📞 Contact

For questions or collaboration opportunities, please open an issue in the repository.

## 🙏 Acknowledgments

- Dataset: [UCI Machine Learning Repository / Kaggle]
- SHAP library for interpretability framework
- Streamlit for dashboard framework
- scikit-learn for ML infrastructure

---

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py

# Run notebooks
jupyter notebook

# Access dashboard
# Open browser to http://localhost:8501
```

## 📊 Dashboard Preview

The dashboard includes:
- **Risk Calculator**: Input patient data → Get instant risk prediction
- **SHAP Explanation**: Visual breakdown of feature contributions
- **Model Comparison**: Side-by-side performance metrics
- **Data Insights**: Interactive visualizations

---

**Built with ❤️ for Healthcare AI**

*Last Updated: December 2024*
*Version: 1.0.0*


