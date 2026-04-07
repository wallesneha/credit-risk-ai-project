import pandas as pd

def load_data():
    print("Loading dataset...")
    df = pd.read_csv("data/loan_data.csv")
    return df

def main():
    df = load_data()
    print(df.head())
    print(df.info())

if __name__ == "__main__":
    main()