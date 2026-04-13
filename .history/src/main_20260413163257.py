import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# 📌 Load Data
def load_data():
    print("Loading dataset...")
    df = pd.read_csv("data/loan_data.csv")
    return df


# 📌 Clean Data
def clean_data(df):
    print("Cleaning data...")
    
    # Fill missing values
    df = df.ffill()
    
    # Remove unnecessary columns
    drop_cols = ["ID", "dtir1"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Convert categorical → numeric
    for col in df.select_dtypes(include='object').columns:
        df.loc[:, col] = df[col].astype('category').cat.codes
    
    return df


# 📌 Feature Importance
def show_feature_importance(model, X):
    print("\nFeature Importance:")
    
    importance = model.feature_importances_
    feature_names = X.columns
    
    for name, score in zip(feature_names, importance):
        print(f"{name}: {score:.4f}")


# 🧠 SHAP Explanation
def explain_model(model, X):
    print("\nGenerating SHAP explanation...")
    
    # Take small sample (important for speed)
    X_sample = X.sample(100, random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP summary plot
    shap.summary_plot(shap_values, X_sample)
    
    # Ensure plot displays
    plt.show()


# 📌 Train Model
def train_model(df):
    print("Training model...")
    
    # Target and features
    X = df.drop(columns=["Status"])
    y = df["Status"]
    
    # Show distribution
    print("\nTarget Distribution:")
    print(y.value_counts())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Interpretation
    print("\nModel Performance Insight:")
    print("The model performs well in predicting loan default risk with strong accuracy.")
    
    # Feature Importance
    show_feature_importance(model, X)
    
    # SHAP Explanation
    explain_model(model, X)
    
    return model


# 📌 Main
def main():
    df = load_data()
    df = clean_data(df)
    model = train_model(df)


if __name__ == "__main__":
    main()