# SCRATCH: my first tuning check (row split). Superseded by notebook Sections 13c, 15 and 16, which also group by order. Note: this script also scores the "best CV" settings on the test set, which the notebook does for reporting only. Not needed to reproduce the results.
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin1')
d = df.drop(columns=['Product Description', 'Order Zipcode']).dropna(subset=['Customer Lname', 'Customer Zipcode'])
target = 'Late_delivery_risk'
feats = ['Shipping Mode', 'Order Region', 'Market', 'Customer Segment', 'Department Name',
         'Category Name', 'Type', 'Order Item Quantity', 'Order Item Discount Rate', 'Sales']
X = d[feats].copy(); y = d[target].copy()
cat = X.select_dtypes(include=['object', 'str']).columns.tolist()
num = [c for c in feats if c not in cat]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ---------- Logistic Regression: regularization strength C ----------
pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat), ('num', StandardScaler(), num)])
print("LOGISTIC REGRESSION (3-fold CV on training split, AUC)")
lr_res = {}
for C in [0.01, 0.1, 1, 10]:
    t = time.time()
    pipe = Pipeline([('pre', pre), ('clf', LogisticRegression(C=C, max_iter=1000))])
    s = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring='roc_auc')
    lr_res[C] = s.mean()
    print(f"  C={C:<5} AUC={s.mean():.4f} (+/- {s.std():.4f})  {time.time()-t:.0f}s", flush=True)

# ---------- Random Forest: depth ----------
Xe = X.copy()
for c in cat:
    Xe[c] = LabelEncoder().fit_transform(Xe[c].astype(str))
Xetr, Xete, yetr, yete = train_test_split(Xe, y, test_size=0.2, random_state=42, stratify=y)
print("\nRANDOM FOREST (3-fold CV on training split, AUC)")
rf_res = {}
for depth in [5, 10, 15]:
    t = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42, n_jobs=-1)
    s = cross_val_score(rf, Xetr, yetr, cv=cv, scoring='roc_auc')
    rf_res[depth] = s.mean()
    print(f"  max_depth={depth:<3} AUC={s.mean():.4f} (+/- {s.std():.4f})  {time.time()-t:.0f}s", flush=True)

# ---------- Held-out test: current settings vs best CV settings ----------
print("\nHELD-OUT TEST SET")
bestC = max(lr_res, key=lr_res.get)
bestD = max(rf_res, key=rf_res.get)
for label, C in [("LR current (C=1)", 1), (f"LR best CV (C={bestC})", bestC)]:
    p = Pipeline([('pre', pre), ('clf', LogisticRegression(C=C, max_iter=1000))]).fit(Xtr, ytr)
    pr = p.predict_proba(Xte)[:, 1]
    print(f"  {label}: acc={accuracy_score(yte, pr>0.5):.4f} AUC={roc_auc_score(yte, pr):.4f}")
for label, dep in [("RF current (depth=10)", 10), (f"RF best CV (depth={bestD})", bestD)]:
    m = RandomForestClassifier(n_estimators=100, max_depth=dep, random_state=42, n_jobs=-1).fit(Xetr, yetr)
    pr = m.predict_proba(Xete)[:, 1]
    print(f"  {label}: acc={accuracy_score(yete, pr>0.5):.4f} AUC={roc_auc_score(yete, pr):.4f}")

# ---------- Threshold check on the logistic regression ----------
print(f"\nClass balance: {y.mean()*100:.1f}% late")
p = Pipeline([('pre', pre), ('clf', LogisticRegression(C=bestC, max_iter=1000))]).fit(Xtr, ytr)
pr = p.predict_proba(Xte)[:, 1]
print("THRESHOLD SWEEP (LR, test accuracy)")
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    print(f"  threshold={th}: acc={accuracy_score(yte, pr>th):.4f}")
