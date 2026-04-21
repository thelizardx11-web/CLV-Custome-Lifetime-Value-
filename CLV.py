# Import libararies 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# load data 
df = pd.read_csv('customer_data.csv')

# Basic exploration 
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


# KEY INSIGHTS
print(f"Total customers: {df['ID'].nunique()}")
print(f"Total records: {len(df)}")
print(f"Date range: {df['Dt_Customer'].min()} to {df['Dt_Customer'].max()}")

# Total spending per order (all product categories)
df['order_amount'] = (
    df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
    df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
)

print(f"Average order value: ₹{df['order_amount'].mean():.2f}")

# Cleaning data 
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
# Agar product category column nahi hai, to ek naya bana lo based on spending
df['product_category'] = df[['MntWines','MntFruits','MntMeatProducts',
                             'MntFishProducts','MntSweetProducts','MntGoldProds']].idxmax(axis=1)
df['product_category'].fillna('Unknown', inplace=True)

# Calculate order_amount (sum of all product categories)
df['order_amount'] = (
    df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
    df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
)
df['order_amount'].fillna(df['order_amount'].median(), inplace=True)

# Convert date column to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

# Remove outliers (IQR method)
Q1 = df['order_amount'].quantile(0.25)
Q3 = df['order_amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['order_amount'] >= Q1 - 1.5*IQR) & 
        (df['order_amount'] <= Q3 + 1.5*IQR)]

print("Data after cleaning:")
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

x
# --- FEATURE ENGINEERING FOR CLV PREDICTION ---

# Group by customer
customer_features = df.groupby('ID').agg({
    'ID': 'count',   # just to keep structure, will rename later
    'order_amount': ['sum', 'mean', 'std'],  # spending metrics (we'll calculate order_amount first)
    'Dt_Customer': ['min', 'max'],           # customer tenure
    'product_category': 'nunique'            # category diversity
}).reset_index(drop=True)

# Flatten column names
customer_features.columns = [
    'customer_id', 'order_count', 'total_revenue',
    'avg_order_value', 'order_std',
    'first_purchase', 'last_purchase',
    'category_diversity'
]

# --- CREATE NEW FEATURES ---

# 1. Customer lifetime value (target variable)
customer_features['clv'] = customer_features['total_revenue']

# 2. Customer tenure (days)
customer_features['tenure_days'] = (
    customer_features['last_purchase'] - customer_features['first_purchase']
).dt.days

# 3. Days since last purchase
today = pd.Timestamp.now()
customer_features['days_since_purchase'] = (
    today - customer_features['last_purchase']
).dt.days

# 4. Purchase frequency (orders per month)
customer_features['purchase_frequency'] = (
    customer_features['order_count'] /
    (customer_features['tenure_days'] / 30 + 1)  # +1 to avoid division by zero
)

# 5. Order value volatility
customer_features['order_volatility'] = (
    customer_features['order_std'] /
    (customer_features['avg_order_value'] + 1)
)

# 6. Customer segment based on CLV
customer_features['clv_segment'] = pd.cut(
    customer_features['clv'],
    bins=[-1, 500, 2000, 5000, float('inf')],
    labels=['Low', 'Medium', 'High', 'VIP']
)

print(customer_features.head())
