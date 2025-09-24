"""
Insights generation utilities for StyleHive dashboard
Provides business insights, KPIs, and analytical summaries
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BusinessInsights:
    def __init__(self, data_preprocessor):
        self.dp = data_preprocessor
        self.df = data_preprocessor.df
        self.product_stats = data_preprocessor.get_product_statistics()
        self.co_occurrence = data_preprocessor.get_co_occurrence_matrix()
        
    def get_kpis(self) -> Dict:
        """Calculate key performance indicators"""
        total_transactions = len(self.df)
        unique_customers = self.df['CustomerID'].nunique()
        unique_products = self.df['Product'].nunique()
        
        # Calculate average basket size
        baskets = self.df.groupby(['CustomerID', 'Date']).size()
        avg_basket_size = baskets.mean()
        
        # Calculate revenue metrics (assuming average prices)
        price_mapping = {
            'White T-shirt': 25, 'Blue Jeans': 60, 'Sneakers': 80, 'Leather Jacket': 200,
            'Sunglasses': 50, 'Backpack': 40, 'Hoodie': 45, 'Formal Shirt': 70,
            'Dress Shoes': 120, 'Smartwatch': 300
        }
        
        self.df['Price'] = self.df['Product'].map(price_mapping)
        total_revenue = self.df['Price'].sum()
        avg_order_value = total_revenue / self.df.groupby(['CustomerID', 'Date']).ngroups
        
        return {
            'total_transactions': total_transactions,
            'unique_customers': unique_customers,
            'unique_products': unique_products,
            'avg_basket_size': round(avg_basket_size, 2),
            'total_revenue': round(total_revenue, 2),
            'avg_order_value': round(avg_order_value, 2),
            'date_range': f"{self.df['Date'].min().strftime('%Y-%m-%d')} to {self.df['Date'].max().strftime('%Y-%m-%d')}"
        }
    
    def get_top_products(self, top_n: int = 5) -> pd.DataFrame:
        """Get top selling products"""
        return self.product_stats.head(top_n)[['Product', 'TotalPurchases', 'UniqueCustomers']]
    
    def get_co_purchase_insights(self) -> List[Dict]:
        """Generate co-purchase insights"""
        insights = []
        
        # Find strongest co-purchases
        co_occurrence_copy = self.co_occurrence.copy()
        np.fill_diagonal(co_occurrence_copy.values, 0)  # Remove self-purchases
        
        # Get top co-purchases
        for product1 in co_occurrence_copy.index:
            for product2 in co_occurrence_copy.columns:
                if product1 != product2:
                    co_purchases = co_occurrence_copy.loc[product1, product2]
                    if co_purchases > 0:
                        # Calculate percentage
                        total_purchases_1 = self.df[self.df['Product'] == product1].shape[0]
                        percentage = (co_purchases / total_purchases_1) * 100
                        
                        if percentage > 20:  # Only include significant co-purchases
                            insights.append({
                                'product1': product1,
                                'product2': product2,
                                'co_purchases': int(co_purchases),
                                'percentage': round(percentage, 1),
                                'insight': f"{percentage:.1f}% of customers who bought {product1} also bought {product2}"
                            })
        
        # Sort by percentage
        insights.sort(key=lambda x: x['percentage'], reverse=True)
        return insights[:10]  # Top 10 insights
    
    def get_seasonal_insights(self) -> Dict:
        """Analyze seasonal purchasing patterns"""
        seasonal_data = self.df.groupby(['Season', 'Product']).size().reset_index(name='Purchases')
        
        # Find seasonal champions
        seasonal_champions = {}
        for season in ['Spring', 'Summer', 'Fall', 'Winter']:
            season_data = seasonal_data[seasonal_data['Season'] == season]
            if len(season_data) > 0:
                top_product = season_data.loc[season_data['Purchases'].idxmax()]
                seasonal_champions[season] = {
                    'product': top_product['Product'],
                    'purchases': int(top_product['Purchases'])
                }
        
        # Calculate seasonal trends
        seasonal_trends = {}
        for product in self.df['Product'].unique():
            product_data = seasonal_data[seasonal_data['Product'] == product]
            if len(product_data) > 0:
                seasonal_trends[product] = {
                    'spring': product_data[product_data['Season'] == 'Spring']['Purchases'].sum(),
                    'summer': product_data[product_data['Season'] == 'Summer']['Purchases'].sum(),
                    'fall': product_data[product_data['Season'] == 'Fall']['Purchases'].sum(),
                    'winter': product_data[product_data['Season'] == 'Winter']['Purchases'].sum()
                }
        
        return {
            'seasonal_champions': seasonal_champions,
            'seasonal_trends': seasonal_trends
        }
    
    def get_customer_segments(self) -> Dict:
        """Analyze customer behavior segments"""
        customer_stats = self.df.groupby('CustomerID').agg({
            'Product': 'count',
            'Date': 'nunique'
        }).rename(columns={'Product': 'TotalPurchases', 'Date': 'ActiveDays'})
        
        # Calculate purchase frequency
        customer_stats['AvgPurchasesPerDay'] = customer_stats['TotalPurchases'] / customer_stats['ActiveDays']
        
        # Define segments
        def categorize_customer(row):
            if row['TotalPurchases'] >= 10 and row['AvgPurchasesPerDay'] >= 0.5:
                return 'High Value'
            elif row['TotalPurchases'] >= 5:
                return 'Medium Value'
            else:
                return 'Low Value'
        
        customer_stats['Segment'] = customer_stats.apply(categorize_customer, axis=1)
        
        segment_summary = customer_stats['Segment'].value_counts().to_dict()
        
        return {
            'segment_distribution': segment_summary,
            'customer_stats': customer_stats
        }
    
    def get_product_affinity_network(self) -> Dict:
        """Create product affinity network data"""
        # Get top co-purchases for network visualization
        network_data = []
        
        for product1 in self.co_occurrence.index:
            for product2 in self.co_occurrence.columns:
                if product1 != product2:
                    co_purchases = self.co_occurrence.loc[product1, product2]
                    if co_purchases > 5:  # Minimum threshold
                        network_data.append({
                            'source': product1,
                            'target': product2,
                            'weight': co_purchases,
                            'strength': co_purchases / self.df[self.df['Product'] == product1].shape[0]
                        })
        
        # Sort by strength
        network_data.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'nodes': list(self.df['Product'].unique()),
            'edges': network_data[:20]  # Top 20 connections
        }
    
    def get_business_recommendations(self) -> List[str]:
        """Generate business recommendations based on data analysis"""
        recommendations = []
        
        # Analyze top products
        top_products = self.product_stats.head(3)['Product'].tolist()
        recommendations.append(f"Focus on promoting {', '.join(top_products)} as they are your top sellers")
        
        # Analyze co-purchases
        co_purchase_insights = self.get_co_purchase_insights()
        if co_purchase_insights:
            top_co_purchase = co_purchase_insights[0]
            recommendations.append(
                f"Bundle {top_co_purchase['product1']} with {top_co_purchase['product2']} "
                f"({top_co_purchase['percentage']}% co-purchase rate)"
            )
        
        # Analyze seasonal patterns
        seasonal_insights = self.get_seasonal_insights()
        if seasonal_insights['seasonal_champions']:
            summer_champion = seasonal_insights['seasonal_champions'].get('Summer', {})
            if summer_champion:
                recommendations.append(
                    f"Promote {summer_champion['product']} during summer months "
                    f"({summer_champion['purchases']} summer purchases)"
                )
        
        # Analyze customer segments
        customer_segments = self.get_customer_segments()
        high_value_pct = (customer_segments['segment_distribution'].get('High Value', 0) / 
                         sum(customer_segments['segment_distribution'].values())) * 100
        recommendations.append(
            f"Focus on customer retention - {high_value_pct:.1f}% are high-value customers"
        )
        
        return recommendations
    
    def get_dashboard_summary(self) -> Dict:
        """Get comprehensive dashboard summary"""
        return {
            'kpis': self.get_kpis(),
            'top_products': self.get_top_products(),
            'co_purchase_insights': self.get_co_purchase_insights()[:5],
            'seasonal_insights': self.get_seasonal_insights(),
            'customer_segments': self.get_customer_segments(),
            'business_recommendations': self.get_business_recommendations()
        }

if __name__ == "__main__":
    # Test the insights
    from data_prep import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    insights = BusinessInsights(preprocessor)
    
    print("KPIs:")
    kpis = insights.get_kpis()
    for key, value in kpis.items():
        print(f"{key}: {value}")
    
    print("\nTop Products:")
    print(insights.get_top_products())
    
    print("\nCo-purchase Insights:")
    co_purchases = insights.get_co_purchase_insights()
    for insight in co_purchases[:3]:
        print(insight['insight'])
    
    print("\nBusiness Recommendations:")
    for rec in insights.get_business_recommendations():
        print(f"- {rec}")
