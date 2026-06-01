import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
n = 100000

start_date = datetime(2025, 1, 1, 0, 0, 0)
minutes_offset = np.random.randint(0, 525600, n)
timestamp = [start_date + timedelta(minutes=int(m)) for m in minutes_offset]

hour_of_day = np.array([t.hour for t in timestamp])
day_of_week = np.array([t.weekday() for t in timestamp])

amount = np.random.lognormal(mean=4.0, sigma=1.2, size=n).clip(1, 25000)

merchant_category = np.random.choice(
    ['grocery', 'dining', 'retail', 'travel', 'entertainment',
     'utilities', 'healthcare', 'gas'],
    size=n,
    p=[0.20, 0.15, 0.18, 0.08, 0.10, 0.12, 0.07, 0.10]
)

account_balance = np.random.lognormal(mean=7.0, sigma=1.5, size=n).clip(50, 100000)

is_new_merchant = np.random.binomial(1, 0.15, n)

distance_from_home = np.random.exponential(scale=20, size=n).clip(0, 3000)

is_international = np.random.binomial(1, 0.08, n)

is_card_present = np.random.binomial(1, 0.85, n)

txns_last_24h = np.random.poisson(2, n)

round_amount_flag = np.array([
    1 if abs(a - round(a / 100) * 100) < 0.01 or abs(a - round(a / 500) * 500) < 0.01
    else 0 for a in amount
])

amount_to_balance = amount / account_balance.clip(1)

odd_hour = ((hour_of_day >= 1) & (hour_of_day <= 5)).astype(int)

logit = (
    + 4.0 * odd_hour
    + 3.0 * is_international
    + 2.5 * is_new_merchant
    + 3.5 * np.clip((amount_to_balance - 3) / 3, 0, 1)
    + 1.5 * np.clip((txns_last_24h - 5) / 5, 0, 1)
    + 2.0 * np.clip((distance_from_home - 500) / 500, 0, 1)
    - 2.5 * is_card_present
    + 1.5 * round_amount_flag
    - 5.5
)

prob_fraud = 1 / (1 + np.exp(-logit))
is_fraud = np.random.binomial(1, prob_fraud)

df = pd.DataFrame({
    'timestamp': timestamp,
    'amount': amount.round(2),
    'merchant_category': merchant_category,
    'account_balance': account_balance.round(2),
    'is_new_merchant': is_new_merchant,
    'distance_from_home': distance_from_home.round(1),
    'is_international': is_international,
    'is_card_present': is_card_present,
    'txns_last_24h': txns_last_24h,
    'round_amount_flag': round_amount_flag,
    'amount_to_balance_ratio': amount_to_balance.round(3),
    'hour_of_day': hour_of_day,
    'day_of_week': day_of_week,
    'is_fraud': is_fraud,
})

df.to_csv('fraud_data.csv', index=False)
print(f'Generated {len(df)} transactions')
print(f'Fraud rate: {df.is_fraud.mean():.2%}')
print(f'Aggregated features: hour_of_day, day_of_week, amount_to_balance_ratio, round_amount_flag')
print(f'Raw features: amount, merchant_category, account_balance, is_new_merchant, distance_from_home, is_international, is_card_present, txns_last_24h')
