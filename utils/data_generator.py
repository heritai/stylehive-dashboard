"""
Synthetic data generator for StyleHive fashion retailer
Creates realistic transaction patterns with co-purchases and seasonal trends
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Tuple

class StyleHiveDataGenerator:
    def __init__(self):
        self.products = [
            "White T-shirt",
            "Blue Jeans", 
            "Sneakers",
            "Leather Jacket",
            "Sunglasses",
            "Backpack",
            "Hoodie",
            "Formal Shirt",
            "Dress Shoes",
            "Smartwatch"
        ]
        
        # Product categories and characteristics
        self.product_info = {
            "White T-shirt": {"category": "basics", "price_tier": "low", "frequency": "high"},
            "Blue Jeans": {"category": "basics", "price_tier": "medium", "frequency": "high"},
            "Sneakers": {"category": "footwear", "price_tier": "medium", "frequency": "high"},
            "Leather Jacket": {"category": "outerwear", "price_tier": "high", "frequency": "low"},
            "Sunglasses": {"category": "accessories", "price_tier": "medium", "frequency": "medium"},
            "Backpack": {"category": "accessories", "price_tier": "medium", "frequency": "medium"},
            "Hoodie": {"category": "casual", "price_tier": "medium", "frequency": "high"},
            "Formal Shirt": {"category": "formal", "price_tier": "medium", "frequency": "medium"},
            "Dress Shoes": {"category": "footwear", "price_tier": "high", "frequency": "low"},
            "Smartwatch": {"category": "tech", "price_tier": "high", "frequency": "low"}
        }
        
        # Strong co-purchase patterns
        self.co_purchase_patterns = {
            "White T-shirt": ["Blue Jeans", "Sneakers", "Hoodie"],
            "Blue Jeans": ["White T-shirt", "Sneakers", "Hoodie"],
            "Sneakers": ["White T-shirt", "Blue Jeans", "Smartwatch"],
            "Leather Jacket": ["Sunglasses", "Dress Shoes"],
            "Sunglasses": ["Leather Jacket", "Backpack"],
            "Backpack": ["Sunglasses", "Sneakers"],
            "Hoodie": ["White T-shirt", "Blue Jeans"],
            "Formal Shirt": ["Dress Shoes", "Blue Jeans"],
            "Dress Shoes": ["Formal Shirt", "Leather Jacket"],
            "Smartwatch": ["Sneakers", "Backpack"]
        }
        
        # Seasonal patterns
        self.seasonal_patterns = {
            "spring": ["White T-shirt", "Blue Jeans", "Sneakers", "Sunglasses"],
            "summer": ["White T-shirt", "Blue Jeans", "Sneakers", "Sunglasses", "Backpack"],
            "fall": ["Hoodie", "Blue Jeans", "Sneakers", "Leather Jacket"],
            "winter": ["Hoodie", "Leather Jacket", "Backpack", "Dress Shoes"]
        }
    
    def get_season(self, date: datetime) -> str:
        """Determine season based on date"""
        month = date.month
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        else:
            return "winter"
    
    def generate_customer_transaction(self, customer_id: int, date: datetime) -> List[str]:
        """Generate a realistic transaction for a customer on a given date"""
        season = self.get_season(date)
        transaction = []
        
        # Base probability of buying each product
        base_probabilities = {
            "White T-shirt": 0.4,
            "Blue Jeans": 0.35,
            "Sneakers": 0.3,
            "Leather Jacket": 0.05,
            "Sunglasses": 0.2,
            "Backpack": 0.15,
            "Hoodie": 0.25,
            "Formal Shirt": 0.1,
            "Dress Shoes": 0.08,
            "Smartwatch": 0.06
        }
        
        # Adjust probabilities based on season
        seasonal_boost = {
            "spring": {"White T-shirt": 1.5, "Sunglasses": 1.8},
            "summer": {"White T-shirt": 2.0, "Sunglasses": 2.2, "Backpack": 1.5},
            "fall": {"Hoodie": 2.0, "Leather Jacket": 1.8},
            "winter": {"Hoodie": 1.8, "Leather Jacket": 2.5, "Backpack": 1.3}
        }
        
        # Apply seasonal adjustments
        for product in self.products:
            prob = base_probabilities[product]
            if product in seasonal_boost[season]:
                prob *= seasonal_boost[season][product]
            
            if random.random() < prob:
                transaction.append(product)
        
        # Add co-purchase items based on what's already in the basket
        co_purchases = []
        for product in transaction:
            if product in self.co_purchase_patterns:
                for co_product in self.co_purchase_patterns[product]:
                    if co_product not in transaction and random.random() < 0.3:
                        co_purchases.append(co_product)
        
        transaction.extend(co_purchases)
        
        # Remove duplicates and return
        return list(set(transaction))
    
    def generate_dataset(self, start_date: str = "2023-01-01", months: int = 18) -> pd.DataFrame:
        """Generate the complete dataset"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = start + timedelta(days=months * 30)
        
        transactions = []
        customer_id = 2001
        
        # Generate transactions for each day
        current_date = start
        while current_date < end:
            # Number of customers shopping each day (varies by day of week)
            if current_date.weekday() < 5:  # Weekday
                daily_customers = random.randint(15, 25)
            else:  # Weekend
                daily_customers = random.randint(25, 40)
            
            for _ in range(daily_customers):
                customer_transaction = self.generate_customer_transaction(customer_id, current_date)
                
                # Add each product as a separate row
                for product in customer_transaction:
                    transactions.append({
                        "CustomerID": customer_id,
                        "Product": product,
                        "Date": current_date.strftime("%Y-%m-%d")
                    })
                
                customer_id += 1
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(transactions)
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "sample_data/stylehive_transactions.csv"):
        """Save dataset to CSV"""
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        print(f"Total transactions: {len(df)}")
        print(f"Unique customers: {df['CustomerID'].nunique()}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

if __name__ == "__main__":
    generator = StyleHiveDataGenerator()
    df = generator.generate_dataset()
    generator.save_dataset(df)
