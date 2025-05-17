import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(data_path):
    data = pd.read_csv(data_path)

    label_encoders = {}
    categorical_columns = ["Gender", "ExtraCurriculars", "Placed"]
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    numeric_columns = ["CGPA", "Backlogs", "Programming_Skills", "Internships", "Projects"]
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    X = data.drop(["StudentID", "Placed"], axis=1)
    y = data["Placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, label_encoders, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess("../data/placement_data.csv")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
