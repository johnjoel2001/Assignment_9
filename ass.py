# advanced_diabetes_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

# Separate features and labels
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Feature engineering: Adding interaction features
X['BMI_Age'] = X['BMI'] * X['Age']
X['Glucose_BloodPressure'] = X['Glucose'] * X['BloodPressure']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing and modeling pipeline
# Handling missing values, scaling, and model training
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
            ('scaler', StandardScaler())  # Scale numerical features
        ]), X.columns)
    ]
)

# Initialize model pipeline with a classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10]
}

# Use GridSearchCV for hyperparameter tuning with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Select the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")
print(f"Model F1 Score: {f1:.2f}")
print(f"Model ROC AUC: {roc_auc:.2f}")

# Save the best model
joblib.dump(best_model, 'diabetes_model_advanced.pkl')
