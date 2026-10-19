"""Rebuilds every chart used in the final deck from the data. Run from the repo root: python scripts/charts.py
Reads data/DataCoSupplyChainDataset.csv; writes PNGs into charts/. Nothing is typed in by hand."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import os
os.makedirs('charts', exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'
CL = {'Standard Class': '#4C72B0', 'Second Class': '#55A868', 'First Class': '#C44E52', 'Same Day': '#8172B2'}
INK, GREY = '#14213D', '#B8BCC4'
modes = ['Same Day', 'First Class', 'Second Class', 'Standard Class']

# ---- data (same cleaning as the notebook) ----
df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin1')
d = df.drop(columns=['Product Description', 'Order Zipcode']).dropna(subset=['Customer Lname', 'Customer Zipcode'])
y = d['Late_delivery_risk']
g = d.groupby('Shipping Mode')
prom = [round(g['Days for shipment (scheduled)'].mean()[m]) for m in modes]
act = [g['Days for shipping (real)'].mean()[m] for m in modes]
rate = {m: g['Late_delivery_risk'].mean()[m] * 100 for m in modes}
n_rows = {m: int(g.size()[m]) for m in modes}
late = d[y == 1]
sales_late = {m: late[late['Shipping Mode'] == m]['Sales'].sum() / 1e6 for m in modes}
overall = y.mean() * 100
n_late_orders = int(d.groupby('Order Id')['Late_delivery_risk'].max().sum())
rule_acc = (d['Shipping Mode'].isin(['First Class', 'Second Class']).astype(int) == y).mean() * 100

# model accuracy on the notebook's 80/20 stratified split
feats = ['Shipping Mode', 'Order Region', 'Market', 'Customer Segment', 'Department Name', 'Category Name',
         'Type', 'Order Item Quantity', 'Order Item Discount Rate', 'Sales']
X = d[feats]; cat = [c for c in feats if X[c].dtype == object or str(X[c].dtype) == 'str']; num = [c for c in feats if c not in cat]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
lr = Pipeline([('p', ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore'), cat), ('n', StandardScaler(), num)])),
               ('m', LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
model_acc = (lr.predict(Xte) == yte).mean() * 100

def cramers(col):
    ct = pd.crosstab(d[col], y); chi2 = chi2_contingency(ct)[0]
    return float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
V = {c: cramers(c) for c in ['Shipping Mode', 'Type', 'Order Region', 'Category Name', 'Market', 'Department Name', 'Customer Segment']}
V_names = {'Type': 'Payment Type'}

def save(name): plt.tight_layout(); plt.savefig(f'charts/{name}.png', dpi=170); plt.close()
def clean(ax): ax.spines[['top', 'right']].set_visible(False)

# A promised days
fig, ax = plt.subplots(figsize=(11, 4.6))
b = ax.bar(modes, prom, color=[CL[m] for m in modes], width=0.55)
for r, v in zip(b, prom): ax.text(r.get_x() + r.get_width() / 2, v + 0.12, f"{v} day{'s' if v != 1 else ''}", ha='center', fontsize=16, fontweight='bold', color=INK)
ax.set_ylim(0, 5); ax.set_ylabel('Days promised', fontsize=13); clean(ax); ax.tick_params(labelsize=14); save('promised')

# B promised vs actual
fig, ax = plt.subplots(figsize=(11, 4.8)); x = np.arange(4); w = 0.36
b1 = ax.bar(x - w / 2, prom, w, color=GREY); b2 = ax.bar(x + w / 2, act, w, color=[CL[m] for m in modes])
for r, v in zip(b1, prom): ax.text(r.get_x() + r.get_width() / 2, v + 0.1, f"{v}", ha='center', fontsize=14, color='#4A5568')
for r, v in zip(b2, act): ax.text(r.get_x() + r.get_width() / 2, v + 0.1, f"{v:.1f}", ha='center', fontsize=14, fontweight='bold', color=INK)
ax.set_xticks(x); ax.set_xticklabels(modes, fontsize=14); ax.set_ylim(0, 5); ax.set_ylabel('Days', fontsize=13); clean(ax)
ax.legend(handles=[Patch(color=GREY, label='Promised'), Patch(color='#4A5568', label='Actual average (tier color)')], frameon=False, fontsize=12, loc='upper left')
save('promised_vs_actual')

# C accuracy ladder
fig, ax = plt.subplots(figsize=(11, 4.4))
lab = ['Guess "late" every time', 'One-line rule: First or Second Class = late', 'Full model, 10 factors']; val = [overall, rule_acc, model_acc]
b = ax.barh(lab[::-1], val[::-1], color=[GREY, '#C44E52', INK][::-1], height=0.55)
for r, v in zip(b, val[::-1]): ax.text(v + 0.8, r.get_y() + r.get_height() / 2, f"{v:.1f}%", va='center', fontsize=17, fontweight='bold', color=INK)
ax.set_xlim(0, 85); ax.set_xlabel('Orders called correctly (%)', fontsize=13); clean(ax); ax.tick_params(labelsize=14); save('ladder')

# D factors (Cramer's V)
rows = sorted(V.items(), key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(11, 4.8))
b = ax.barh([V_names.get(n, n) for n, _ in rows], [v for _, v in rows], color=['#C44E52' if n == 'Shipping Mode' else GREY for n, _ in rows], height=0.62)
for r, (n, v) in zip(b, rows): ax.text(v + 0.008, r.get_y() + r.get_height() / 2, f"{v:.2f}", va='center', fontsize=14, fontweight='bold' if n == 'Shipping Mode' else 'normal', color=INK)
ax.set_xlim(0, 0.55); ax.set_xlabel('How closely each factor tracks late delivery (0 = not at all, 1 = perfectly)', fontsize=12); clean(ax); ax.tick_params(labelsize=14); save('factors')

# E ceiling vs handling cost (handling cost per unique late order)
ceiling = late['Sales'].sum() / 1e6
vals = [ceiling] + [n_late_orders * r / 1e6 for r in (40, 20, 15)]
lab = ['Revenue on late orders (ceiling)'] + [f'Handling cost if ${r} per late order' for r in (40, 20, 15)]
fig, ax = plt.subplots(figsize=(11, 4.4))
b = ax.barh(lab[::-1], vals[::-1], color=[INK, '#C44E52', '#C44E52', '#C44E52'][::-1], height=0.55)
for r, v in zip(b, vals[::-1]): ax.text(v + 0.3, r.get_y() + r.get_height() / 2, f"${v:.2f}M", va='center', fontsize=16, fontweight='bold', color=INK)
ax.set_xlim(0, 24); ax.set_xlabel('$ millions', fontsize=13); clean(ax); ax.tick_params(labelsize=13); save('worth')

# F late rate by shipping mode (slide 4). Drawn 16 x 5.6 in so it fills the 1664 x 560 px slide box.
order = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
fig, ax = plt.subplots(figsize=(16, 5.6))
b = ax.bar(order, [rate[m] for m in order], color=[CL[m] for m in order], width=0.6)
ax.axhline(overall, color='#555555', ls='--', lw=1.6); ax.text(3.45, overall + 1.8, f'Overall: {overall:.1f}%', ha='right', fontsize=18, color='#555555')
for r, m in zip(b, order):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{rate[m]:.1f}%", ha='center', fontsize=22, fontweight='bold', color=INK)
    ax.text(r.get_x() + r.get_width() / 2, 3, f"{n_rows[m]:,} items", ha="center", fontsize=18, color='white')
ax.set_ylim(0, 110); ax.set_ylabel('Late (%)', fontsize=18); clean(ax); ax.tick_params(labelsize=18); save('rates')

# G rate vs dollars (slide 6), same 16 x 5.6 box
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
ax = axes[0]; b = ax.bar(order, [rate[m] for m in order], color=[CL[m] for m in order], width=0.6)
for r, m in zip(b, order): ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{rate[m]:.0f}%", ha='center', fontsize=20, fontweight='bold', color=INK)
ax.set_ylim(0, 112); ax.set_title('How often it is late', fontsize=22, fontweight='bold', color=INK); clean(ax); ax.tick_params(labelsize=15)
ax = axes[1]; b = ax.bar(order, [sales_late[m] for m in order], color=[CL[m] for m in order], width=0.6)
for r, m in zip(b, order): ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.15, f"${sales_late[m]:.1f}M", ha='center', fontsize=20, fontweight='bold', color=INK)
ax.set_ylim(0, 10.5); ax.set_title('Sales on late orders ($M)', fontsize=22, fontweight='bold', color=INK); clean(ax); ax.tick_params(labelsize=15)
save('money')
print('overall %.1f rule %.1f model %.1f late orders %d ceiling %.2f' % (overall, rule_acc, model_acc, n_late_orders, ceiling))
print({k: round(v, 4) for k, v in V.items()})
