import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report
from xgboost import XGBClassifier

df = pd.read_csv("data/processed_scada.csv")

features = [
    "WindSpeed",
    "ActualPower",
    "TheoreticalPower",
    "WindDirection",
    "PowerDeviation"
]

X = df[features]
y = df["AtRisk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    eval_metric="logloss"
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
recall = recall_score(y_test, preds)

with open("results/metrics.txt", "w") as f:
    f.write(f"Recall: {recall:.4f}\n")
    f.write(classification_report(y_test, preds))

print("Model trained successfully.")
print(f"Recall: {recall:.4f}")
