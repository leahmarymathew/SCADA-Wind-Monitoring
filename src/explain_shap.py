import pandas as pd
import shap
import matplotlib.pyplot as plt
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

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss"
)

model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, show=False)
plt.savefig("results/shap_summary.png")
plt.close()

print("SHAP explanation generated.")
