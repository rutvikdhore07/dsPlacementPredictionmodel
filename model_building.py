from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            
    return best_model

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess("../data/placement_data.csv")
    best_model = train_and_evaluate(X_train, X_test, y_train, y_test)
