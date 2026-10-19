# SCRATCH: I used this in Week 5 to print the inputs for the late-rate and revenue charts. The final charts are now built by scripts/charts.py, and the same numbers are in notebook Sections 4, 12 and 14. Not needed to reproduce the results.
import pandas as pd

df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin1')
df_clean = df.drop(columns=['Product Description','Order Zipcode']).dropna(subset=['Customer Lname','Customer Zipcode'])

target = 'Late_delivery_risk'

print("=== Chart 1 inputs: Late Delivery Rate by Shipping Mode ===")
g = df_clean.groupby('Shipping Mode')[target].agg(['mean','count'])
g['rate_pct'] = (g['mean']*100).round(1)
print(g)
overall = df_clean[target].mean()*100
print(f"Overall average late rate: {overall:.1f}%")

print()
print("=== Chart 2 inputs: Financial Exposure by Shipping Mode ===")
late = df_clean[df_clean[target]==1]
exposure = late.groupby('Shipping Mode').agg(
    late_orders=('Sales','count'),
    revenue_at_risk=('Sales','sum')
)
exposure['assumed_cost_med_20'] = exposure['late_orders']*20
# join late rate for the "after" chart
exposure = exposure.join(g[['rate_pct','count']])
exposure = exposure.rename(columns={'count':'total_orders_in_mode'})
exposure['revenue_at_risk_per_late_order'] = (exposure['revenue_at_risk']/exposure['late_orders']).round(2)
print(exposure)
print()
print("Sorted by revenue_at_risk desc:")
print(exposure.sort_values('revenue_at_risk', ascending=False)[['revenue_at_risk','rate_pct','late_orders']])
