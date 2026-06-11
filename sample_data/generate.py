# sample_data/generate.py
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1200
dates     = pd.date_range("2021-01-01", periods=n, freq="D")
products  = np.random.choice(["Laptop","Phone","Tablet","Watch","Earbuds"], n)
regions   = np.random.choice(["North","South","East","West"], n)
channels  = np.random.choice(["Online","Retail","Partner"], n)
customers = [f"CUST{str(i).zfill(4)}" for i in np.random.randint(1, 301, n)]
qty       = np.random.randint(1, 15, n)
price     = np.where(products=="Laptop", 75000,
            np.where(products=="Phone",  35000,
            np.where(products=="Tablet", 25000,
            np.where(products=="Watch",  15000, 5000))))
discount  = np.random.uniform(0, 0.25, n).round(2)
revenue   = (qty * price * (1 - discount)).round(2)
csat      = np.random.choice([1,2,3,4,5], n, p=[0.05,0.10,0.20,0.35,0.30])
churn     = np.where((csat <= 2) | (discount < 0.05), 1, 0)
churn     = np.where(np.random.rand(n) < 0.12, 1, churn)

df = pd.DataFrame({
    "date":        dates,
    "customer_id": customers,
    "product":     products,
    "region":      regions,
    "channel":     channels,
    "quantity":    qty,
    "unit_price":  price,
    "discount":    discount,
    "revenue":     revenue,
    "csat_score":  csat,
    "churn":       churn,
})

df.to_csv("sample_data/sample_sales.csv", index=False)
print("✅ Sample dataset created!", df.shape)
print(df.head())