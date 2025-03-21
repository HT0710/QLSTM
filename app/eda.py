import pandas as pd
from rich import print

print()

data = pd.read_csv("qlstm/data/Albany_WA.csv")

data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

df = data.copy()

df["date"] = pd.to_datetime(df["date"])
df["hour"] = pd.to_timedelta(df["hour"], unit="h")
df["date"] = df["date"] + df["hour"]
df = df.rename(columns={"date": "datetime"})
df.drop(["year", "hour", "month", "day"], axis=1, inplace=True)

df.index = df["datetime"]

num_cols = df.select_dtypes(include=["number"]).columns
print(df.columns)
df = df[num_cols]
df.info()
