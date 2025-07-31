import pandas as pd
import pickle
import argparse
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature

def load_data(data_path):
    df = pd.read_csv(data_path)
    X = df[['trip_distance', 'trip_duration']]
    y = df['fare_amount']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions

def log_model_to_mlflow(model, X_train, y_train, mse, tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("mse", mse)
        signature = infer_signature(X_train, model.predict(X_train))
        
        mlflow.sklearn.log_model(
            sk_model=model,
            name="FareEstimationModel",
            signature=signature,
            input_example=X_train,
            registered_model_name="FareEstimator"
        )

def save_model_locally(model, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a fare prediction model.")
    parser.add_argument('--data-path', required=True, help='Path to the CSV dataset')
    parser.add_argument('--model-output', default='lr_model.bin', help='Filename for the saved model')
    parser.add_argument('--tracking-uri', default='http://localhost:5000', help='MLflow tracking URI')
    parser.add_argument('--experiment-name', default='FareEstimationExperiment', help='MLflow experiment name')
    return parser.parse_args()

def main():
    args = parse_args()

    print("Loading data...")
    X, y = load_data(args.data_path)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Training model...")
    model = train_model(X_train, y_train)
    
    print("Evaluating model...")
    mse, _ = evaluate_model(model, X_test, y_test)
    print(f"MSE on test set: {mse:.4f}")
    
    print("Logging model to MLflow...")
    log_model_to_mlflow(model, X_train, y_train, mse, args.tracking_uri, args.experiment_name)
    
    print(f"Saving model locally to {args.model_output}...")
    save_model_locally(model, args.model_output)

    print("Training complete!")

if __name__ == '__main__':
    main()
