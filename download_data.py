# pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=350)
X, y = ds.data.features.copy(), ds.data.targets.copy()

names = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
         "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
         "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
         "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
X.columns = names
X["default"] = y.iloc[:, 0].values
X.to_csv("data/UCI_Credit_Card.csv", index=False)
print(X.shape); print(X.head())