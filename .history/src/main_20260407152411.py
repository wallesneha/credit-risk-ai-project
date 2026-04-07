import pandas as pd

def load_data():
    print("Loading dataset...")
    df = pd.read_csv("data/loan_data.csv")
    return df

def clean_data(df):
    print("Cleaning data...")
    
    # Remove missing values
    df = df.dropna()
    
    # Example: convert categorical to numeric (if exists)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes
    
    return df

def main():
    df = load_data()
    
    print("Before Cleaning:")
    print(df.info())
    
    df = clean_data(df)
    
    print("After Cleaning:")
    print(df.info())
    
    print(df.head())

if __name__ == "__main__":
    main()