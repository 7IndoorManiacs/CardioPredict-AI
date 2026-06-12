import pandas as pd
import kagglehub

path = kagglehub.dataset_download(
    "redwankarimsony/heart-disease-data"
)

df = pd.read_csv(
    f"{path}/heart_disease_uci.csv"
)

columns = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal"
]

for col in columns:
    print("\n" + "="*40)
    print(col.upper())
    print("="*40)
    print(df[col].dropna().unique())