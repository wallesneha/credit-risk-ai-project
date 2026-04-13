import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    print("Loading dataset...")
    df = pd.read_csv("data/loan_data.csv")
    return df

def clean_data(df):
    print("Cleaning data...")
    df = df.dropna()
    
    for col in df.select_dtypes(include='object').columns:
        df.loc[:, col] = df[col].astype('category').cat.codes   # ✅ fixed warning
    
    return df

# ✅ MOVE THIS HERE
def show_feature_importance(model, X):
    print("\nFeature Importance:")
    
    importance = model.feature_importances_
    feature_names = X.columns
    
    for name, score in zip(feature_names, importance):
        print(f"{name}: {score:.4f}")

def train_model(df):
    print("Training model...")
    
    # ✅ FIXED target
    X = df.drop(columns=["Status"])
    y = df["Status"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    show_feature_importance(model, X)
    
    return model

def main():
    df = load_data()
    df = clean_data(df)
    model = train_model(df)

if __name__ == "__main__":
    main()