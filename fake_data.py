# util/fake_data.py
from typing import Optional
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

def generate_realistic_synthetic_dataset(
    num_transactions: int = 5000,
    fraud_percentage: float = 5,
    outlier_percentage: float = 5,
    return_percentage: float = 3,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    product_categories = {
        'Electronics': (300, 2000),
        'Clothing': (20, 300),
        'Groceries': (5, 200),
        'Furniture': (500, 5000),
        'Books': (10, 100),
    }
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    data = []
    start_date = datetime(2023, 1, 1)
    outlier_count = int(num_transactions * outlier_percentage / 100)
    fraud_count = int(num_transactions * fraud_percentage / 100)

    for i in range(num_transactions):
        transaction_id = f"T{i+1:06}"
        timestamp = start_date + timedelta(minutes=int(np.random.poisson(30)))
        customer_id = f"C{random.randint(1, 1000):05}"
        trans_amount = float(max(1.0, np.random.exponential(scale=100.0)))
        product_category = np.random.choice(list(product_categories.keys()))
        location = random.choice(locations)
        fraud_indicator = 0
        is_return = 1 if random.random() < (return_percentage / 100) else 0

        min_amount, max_amount = product_categories[product_category]
        trans_amount = float(np.clip(trans_amount, min_amount, max_amount))

        if i < outlier_count:
            if random.random() < 0.5:
                trans_amount = float(random.uniform(10000, 50000))
            if random.random() < 0.3:
                product_category = "Luxury Items"
            if random.random() < 0.3:
                location = fake.city()

        if i >= num_transactions - fraud_count:
            fraud_indicator = 1

        data.append({
            'TransactionID': transaction_id,
            'Timestamp': timestamp,
            'CustomerID': customer_id,
            'ProductCategory': product_category,
            'TransactionAmount': round(trans_amount, 2),
            'TransactionFrequency': int(np.random.poisson(3)),
            'Location': location,
            'FraudIndicator': fraud_indicator,
            'IsReturn': is_return
        })

    df = pd.DataFrame(data)
    return df
