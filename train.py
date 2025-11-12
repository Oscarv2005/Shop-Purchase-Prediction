import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib, os

DATA = "online_shoppers_intention.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
OUT = os.path.join(MODEL_DIR, "model_joblib.pkl")

df = pd.read_csv(DATA)
y = df["Revenue"].astype(int)
X = df.drop("Revenue", axis=1).copy()

if "Weekend" in X.columns and X["Weekend"].dtype == "bool":
    X["Weekend"] = X["Weekend"].astype(int)

X_proc = pd.get_dummies(X, drop_first=True).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump({"model": clf, "columns": list(X_proc.columns)}, OUT)
print(f"Model saved to {OUT}")
print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))
