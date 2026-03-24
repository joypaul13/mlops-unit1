import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

print("=" * 70)
print("MACHINE LEARNING WORKFLOW - SALARY PREDICTION")
print("=" * 70)

# Step 1: Load the dataset
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('sample_data.csv')
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Step 2: Prepare features and target
print("\n" + "=" * 70)
print("[STEP 2] Preparing features and target variable...")
print("=" * 70)

# Features: age and experience
X = df[['age', 'experience']]
# Target: salary
y = df['salary']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("\nFeatures (first 5 rows):")
print(X.head())
print("\nTarget (first 5 rows):")
print(y.head())

# Step 3: Split data into training and testing sets
print("\n" + "=" * 70)
print("[STEP 3] Splitting data into train and test sets...")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Train-Test split ratio: {len(X_train)/len(X)*100:.1f}% - {len(X_test)/len(X)*100:.1f}%")

# Step 4: Train the Linear Regression model
print("\n" + "=" * 70)
print("[STEP 4] Training Linear Regression model...")
print("=" * 70)

model = LinearRegression()
model.fit(X_train, y_train)

print("✓ Model training completed!")
print(f"\nModel coefficients:")
print(f"  Age coefficient: {model.coef_[0]:.4f}")
print(f"  Experience coefficient: {model.coef_[1]:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# Step 5: Make predictions
print("\n" + "=" * 70)
print("[STEP 5] Making predictions on test set...")
print("=" * 70)

y_pred = model.predict(X_test)

print("Actual vs Predicted values:")
print("-" * 50)
for i in range(len(y_test)):
    print(f"Actual: ${y_test.iloc[i]:,.0f} | Predicted: ${y_pred[i]:,.2f}")

# Step 6: Evaluate the model
print("\n" + "=" * 70)
print("[STEP 6] Model Evaluation Metrics")
print("=" * 70)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"\nInterpretation: The model explains {r2*100:.2f}% of the variance in salary.")

# Step 7: Save the model
print("\n" + "=" * 70)
print("[STEP 7] Saving the trained model...")
print("=" * 70)

model_filename = 'salary_prediction_model.joblib'
joblib.dump(model, model_filename)
print(f"✓ Model saved as '{model_filename}'")

# Step 8: Load and test the saved model
print("\n" + "=" * 70)
print("[STEP 8] Loading and testing the saved model...")
print("=" * 70)

loaded_model = joblib.load(model_filename)
test_prediction = loaded_model.predict([[30, 5]])
print(f"✓ Model loaded successfully!")
print(f"Test prediction: For age=30 and experience=5 years")
print(f"Predicted salary: ${test_prediction[0]:,.2f}")

print("\n" + "=" * 70)
print("ML WORKFLOW COMPLETED SUCCESSFULLY!")
print("=" * 70)
