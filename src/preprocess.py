import pandas as pd

df = pd.read_csv("data/raw_scada.csv")

df = df.rename(columns={
    "LV ActivePower": "ActualPower",
    "Wind Speed": "WindSpeed",
    "Theoretical_Power": "TheoreticalPower",
    "Wind Direction": "WindDirection"
})

df = df.dropna()

df = df[
    (df["WindSpeed"] >= 0) &
    (df["ActualPower"] >= 0) &
    (df["TheoreticalPower"] >= 0)
]

df["PowerDeviation"] = (
    df["TheoreticalPower"] - df["ActualPower"]
) / (df["TheoreticalPower"] + 1e-6)

df["AtRisk"] = 0

deviation_threshold = df["PowerDeviation"].quantile(0.90)

df.loc[
    (df["WindSpeed"] > 5) &
    (df["PowerDeviation"] > deviation_threshold),
    "AtRisk"
] = 1

df.to_csv("data/processed_scada.csv", index=False)

print("Processed data saved successfully.")
