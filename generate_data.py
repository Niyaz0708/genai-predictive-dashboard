import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
n_days = len(dates)

data = {
    "Date": dates,
    "Marketing_Spend": np.random.uniform(500, 5000, n_days),
    "Discount_Percentage": np.random.uniform(0, 25, n_days),
    "Competitor_Price": np.random.uniform(15, 30, n_days),
}

df = pd.DataFrame(data)
df["Sales"] = (df["Marketing_Spend"] * 0.15) - (df["Discount_Percentage"] * 2) + (30 - df["Competitor_Price"]) * 5 + np.random.normal(100, 50, n_days)

df.to_csv("company_sales.csv", index=False)
print("✅ company_sales.csv generated successfully!")
