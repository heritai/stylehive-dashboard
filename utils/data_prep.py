"""
Data preparation utilities for StyleHive recommendation system
Handles data loading, cleaning, and transformation for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_path: str = "sample_data/stylehive_transactions.csv"):
        self.data_path = data_path
        self.df = None
        self.transaction_matrix = None
        self.product_stats = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and basic preprocessing of transaction data"""
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Add derived features
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['DayOfWeek'] = self.df['Date'].dt.day_name()
        self.df['Season'] = self.df['Date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        return self.df
    
    def get_transaction_matrix(self) -> pd.DataFrame:
        """Create customer-product transaction matrix for collaborative filtering"""
        if self.df is None:
            self.load_data()
        
        # Group by customer and product to handle multiple purchases
        customer_products = self.df.groupby(['CustomerID', 'Product']).size().reset_index(name='PurchaseCount')
        
        # Create pivot table
        self.transaction_matrix = customer_products.pivot(
            index='CustomerID', 
            columns='Product', 
            values='PurchaseCount'
        ).fillna(0)
        
        return self.transaction_matrix
    
    def get_basket_data(self) -> List[List[str]]:
        """Get basket data for market basket analysis"""
        if self.df is None:
            self.load_data()
        
        # Group transactions by customer and date to get baskets
        baskets = self.df.groupby(['CustomerID', 'Date'])['Product'].apply(list).tolist()
        
        # Filter out single-item baskets for better association rules
        multi_item_baskets = [basket for basket in baskets if len(basket) > 1]
        
        # Limit to recent data for faster processing (last 6 months)
        if len(multi_item_baskets) > 1000:
            multi_item_baskets = multi_item_baskets[:1000]
        
        return multi_item_baskets
    
    def get_product_statistics(self) -> pd.DataFrame:
        """Calculate product-level statistics"""
        if self.df is None:
            self.load_data()
        
        stats = []
        for product in self.df['Product'].unique():
            product_data = self.df[self.df['Product'] == product]
            
            stats.append({
                'Product': product,
                'TotalPurchases': len(product_data),
                'UniqueCustomers': product_data['CustomerID'].nunique(),
                'AvgPurchasesPerCustomer': len(product_data) / product_data['CustomerID'].nunique(),
                'FirstPurchase': product_data['Date'].min(),
                'LastPurchase': product_data['Date'].max(),
                'PurchaseFrequency': len(product_data) / self.df['Date'].nunique()
            })
        
        self.product_stats = pd.DataFrame(stats).sort_values('TotalPurchases', ascending=False)
        return self.product_stats
    
    def get_co_occurrence_matrix(self) -> pd.DataFrame:
        """Calculate product co-occurrence matrix"""
        if self.df is None:
            self.load_data()
        
        # Get baskets
        baskets = self.df.groupby(['CustomerID', 'Date'])['Product'].apply(list).tolist()
        
        # Create co-occurrence matrix
        products = sorted(self.df['Product'].unique())
        co_occurrence = pd.DataFrame(0, index=products, columns=products)
        
        for basket in baskets:
            if len(basket) > 1:
                for i, product1 in enumerate(basket):
                    for j, product2 in enumerate(basket):
                        if i != j:
                            co_occurrence.loc[product1, product2] += 1
        
        return co_occurrence
    
    def get_customer_insights(self) -> Dict:
        """Calculate customer-level insights"""
        if self.df is None:
            self.load_data()
        
        customer_stats = self.df.groupby('CustomerID').agg({
            'Product': 'count',
            'Date': ['min', 'max', 'nunique']
        }).round(2)
        
        customer_stats.columns = ['TotalPurchases', 'FirstPurchase', 'LastPurchase', 'ActiveDays']
        customer_stats['AvgPurchasesPerDay'] = customer_stats['TotalPurchases'] / customer_stats['ActiveDays']
        
        return {
            'total_customers': len(customer_stats),
            'avg_purchases_per_customer': customer_stats['TotalPurchases'].mean(),
            'avg_basket_size': customer_stats['TotalPurchases'].mean(),
            'most_active_customer': customer_stats['TotalPurchases'].idxmax(),
            'customer_stats': customer_stats
        }
    
    def get_seasonal_patterns(self) -> pd.DataFrame:
        """Analyze seasonal purchasing patterns"""
        if self.df is None:
            self.load_data()
        
        seasonal_data = self.df.groupby(['Season', 'Product']).size().reset_index(name='Purchases')
        seasonal_pivot = seasonal_data.pivot(index='Product', columns='Season', values='Purchases').fillna(0)
        
        # Calculate seasonal percentages
        seasonal_pct = seasonal_pivot.div(seasonal_pivot.sum(axis=1), axis=0) * 100
        
        return seasonal_pct
    
    def get_time_series_data(self) -> pd.DataFrame:
        """Get time series data for trend analysis"""
        if self.df is None:
            self.load_data()
        
        # Daily sales by product
        daily_sales = self.df.groupby(['Date', 'Product']).size().reset_index(name='Sales')
        daily_sales = daily_sales.pivot(index='Date', columns='Product', values='Sales').fillna(0)
        
        return daily_sales
    
    def prepare_for_ml(self) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for machine learning models"""
        if self.df is None:
            self.load_data()
        
        # Get transaction matrix
        transaction_matrix = self.get_transaction_matrix()
        
        # Get product list
        products = list(transaction_matrix.columns)
        
        return transaction_matrix, products

if __name__ == "__main__":
    # Test the data preparation
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Test product statistics
    product_stats = preprocessor.get_product_statistics()
    print("\nTop 5 products by purchases:")
    print(product_stats.head())
    
    # Test co-occurrence
    co_occurrence = preprocessor.get_co_occurrence_matrix()
    print(f"\nCo-occurrence matrix shape: {co_occurrence.shape}")
