# MLOps Unit 1 - Salary Prediction Project

A comprehensive machine learning operations project demonstrating best practices for ML model development, training, evaluation, and deployment.

## 📁 Project Structure

```
mlops-unit1/
├── data/                          # Data directory
│   ├── raw/                      # Raw input data
│   │   └── sample_data.csv
│   └── processed/                # Processed/transformed data
├── src/                          # Source code
│   ├── __init__.py
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── load.py              # Data loading utilities
│   ├── models/                   # Model training and inference
│   │   ├── __init__.py
│   │   ├── ml_model.py          # Complete ML workflow script
│   │   ├── train.py             # Model training logic
│   │   └── predict.py           # Model prediction and loading
│   └── features/                 # Feature engineering
│       └── __init__.py
├── models/                        # Trained model artifacts
│   └── salary_prediction_model.joblib
├── stats.py                       # Data statistics script
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Load Data and Train Model
```bash
python src/models/ml_model.py
```

### 3. Make Predictions
```python
from src.models.predict import load_model, make_predictions

model = load_model('models/salary_prediction_model.joblib')
predictions = make_predictions(model, [[30, 5]])
```

## 📊 Dataset

**Source:** `data/raw/sample_data.csv`

**Features:**
- `age`: Employee age (years)
- `experience`: Years of work experience
- `salary`: Annual salary (target variable)

**Sample Data:**
| name | age | salary | experience |
|------|-----|--------|-----------|
| Alice | 28 | 50000 | 3 |
| Bob | 35 | 65000 | 8 |
| Charlie | 42 | 75000 | 15 |

## 🤖 Model Details

**Algorithm:** Linear Regression

**Model Coefficients:**
- Age: 2401.37
- Experience: -818.14
- Intercept: -13293.38

**Model Performance:**
- Mean Squared Error (MSE): 7,380,001.18
- Root Mean Squared Error (RMSE): 2,716.62
- Mean Absolute Error (MAE): 2,275.07
- R² Score: -0.1808

**Note:** The negative R² score indicates the model's performance on the test set could be improved with more data or feature engineering.

## 📦 Key Modules

### `src/data/load.py`
- `load_raw_data(filepath)`: Load CSV data
- `validate_data(df)`: Validate data quality

### `src/models/train.py`
- `train_model(X, y, test_size, random_state)`: Train Linear Regression model
- `evaluate_model(y_true, y_pred)`: Calculate evaluation metrics

### `src/models/predict.py`
- `load_model(model_path)`: Load trained model
- `save_model(model, model_path)`: Save trained model
- `make_predictions(model, X)`: Generate predictions

## 📝 Scripts

### `stats.py`
Generate comprehensive statistics about the dataset including:
- Dataset overview
- Basic statistics
- Data types
- Missing values
- Descriptive statistics
- Categorical value counts

**Usage:**
```bash
python stats.py
```

### `src/models/ml_model.py`
Complete ML workflow including:
- Data loading
- Feature preparation
- Train-test split
- Model training
- Evaluation metrics
- Model persistence

**Usage:**
```bash
python src/models/ml_model.py
```

##  Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning
- joblib: Model serialization

## 📊 Data Analysis

Run the statistics script for exploratory data analysis:
```bash
python stats.py
```

## 🔄 Version Control

- **Main Branch:** Production-ready code
- **Feature Branches:** Development branches for new features
- **Commits:** Descriptive commit messages for tracking changes

## 🤝 Contributing

When contributing to this project:
1. Create a feature branch
2. Make your changes
3. Run tests
4. Commit with descriptive messages
5. Push and create a pull request

## 📄 License

This project is part of MLOps Unit 1 training material.

## ✅ Checklist

- [x] Data loading and exploration
- [x] Model training pipeline
- [x] Evaluation metrics
- [x] Model persistence
- [x] Project structure
- [ ] CI/CD pipeline
- [ ] Model monitoring
- [ ] Docker containerization

## 📧 Contact

Created by: Joy Paul

## 🎯 Next Steps

1. Improve model performance with feature engineering
2. Set up CI/CD pipeline
3. Add model serving API
4. Implement model monitoring
