# SCRATCH: quick check I used while building the final deck (late rate, sales on late orders, rule accuracy). Superseded by notebook Sections 13 and 14. Not needed to reproduce the results.
import pandas as pd
df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin1')
d = df.drop(columns=['Product Description','Order Zipcode']).dropna(subset=['Customer Lname','Customer Zipcode'])
print(len(df), len(d))
for name,x in [('raw',df),('clean',d)]:
    print(name, x['Late_delivery_risk'].mean())
g = d.groupby('Shipping Mode').agg(n=('Late_delivery_risk','size'), late=('Late_delivery_risk','mean'),
    sched=('Days for shipment (scheduled)','mean'), act=('Days for shipping (real)','mean'), actmed=('Days for shipping (real)','median'))
print(g)
late = d[d.Late_delivery_risk==1]
print('late n', len(late), 'sales', late['Sales'].sum())
print(late.groupby('Shipping Mode')['Sales'].sum(), late.groupby('Shipping Mode').size())
fs = late['Shipping Mode'].isin(['First Class','Second Class']).mean(); print('first+second share of late', fs)
rule = d['Shipping Mode'].isin(['First Class','Second Class']).astype(int)
print('rule acc', (rule==d.Late_delivery_risk).mean(), 'always late', d.Late_delivery_risk.mean())
