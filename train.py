import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import mlflow

# Set up MLflow
mlflow.set_experiment("Wine Quality Prediction")

# Load versioned data
df = pd.read_csv('data/wine_data.csv')
X = df.drop('is_premium', axis=1)
y = df['is_premium']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="GradientBoosting_Baseline"):
    # Define and log parameters
    params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}
    mlflow.log_params(params)

    # Train model
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate and log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model artifact natively
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Model trained with accuracy: {accuracy:.4f}")