# src/main.py

import pandas as pd

def load_data():
    print("Loading dataset...")
    # temporary dummy dataset
    data = {
        "income": [50000, 60000, 30000],
        "loan": [20000, 25000, 15000],
        "default": [0, 0, 1]
    }
    df = pd.DataFrame(data)
    return df

def main():
    df = load_data()
    print(df.head())

if __name__ == "__main__":
    main()