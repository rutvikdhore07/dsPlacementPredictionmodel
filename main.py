from src.data_preprocessing import load_and_preprocess
from src.model_building import train_and_evaluate

def main():
    data_path = "./data/placement_data.csv"
    X_train, X_test, y_train, y_test, label_encoders, scaler = load_and_preprocess(data_path)
    best_model = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("Model training and evaluation completed.")

if __name__ == "__main__":
    main()
