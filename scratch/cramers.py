# SCRATCH: prints Cramer's V and the chi-square p-value for each factor. The notebook does the same in Section 6; scripts/charts.py recomputes V for the factor chart. Not needed to reproduce the results.
import pandas as pd
from scipy.stats import chi2_contingency
df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin1')
d = df.drop(columns=['Product Description','Order Zipcode']).dropna(subset=['Customer Lname','Customer Zipcode'])
for c in ['Shipping Mode','Type','Customer Segment','Market','Order Region','Department Name','Category Name']:
    ct = pd.crosstab(d[c], d['Late_delivery_risk'])
    chi2,p,_,_ = chi2_contingency(ct)
    n = ct.values.sum()
    v = (chi2/(n*(min(ct.shape)-1)))**0.5
    print(f"{c}: V={v:.4f} p={p:.3g} groups={ct.shape[0]}")
