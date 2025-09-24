"""
Recommendation models for StyleHive
Implements Market Basket Analysis and Collaborative Filtering
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class MarketBasketAnalyzer:
    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.1):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = None
        self.rules = None
        
    def fit(self, baskets: List[List[str]]) -> 'MarketBasketAnalyzer':
        """Fit the market basket analysis model"""
        # Convert baskets to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        df_baskets = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        self.frequent_itemsets = apriori(
            df_baskets, 
            min_support=self.min_support, 
            use_colnames=True
        )
        
        # Generate association rules
        if len(self.frequent_itemsets) > 0:
            self.rules = association_rules(
                self.frequent_itemsets, 
                metric="confidence", 
                min_threshold=self.min_confidence
            )
        else:
            self.rules = pd.DataFrame()
            
        return self
    
    def get_recommendations(self, product: str, top_n: int = 5) -> List[Dict]:
        """Get recommendations for a given product"""
        if self.rules is None or len(self.rules) == 0:
            return []
        
        # Find rules where the product is in antecedents
        product_rules = self.rules[
            self.rules['antecedents'].apply(lambda x: product in x)
        ].copy()
        
        if len(product_rules) == 0:
            return []
        
        # Sort by confidence and lift
        product_rules = product_rules.sort_values(['confidence', 'lift'], ascending=False)
        
        recommendations = []
        for _, rule in product_rules.head(top_n).iterrows():
            # Get the consequent product
            consequent = list(rule['consequents'])[0]
            if consequent != product:  # Don't recommend the same product
                recommendations.append({
                    'product': consequent,
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support'],
                    'explanation': f"{rule['confidence']:.1%} of customers who bought {product} also bought {consequent}"
                })
        
        return recommendations
    
    def get_basket_recommendations(self, basket: List[str], top_n: int = 5) -> List[Dict]:
        """Get recommendations for a basket of products"""
        if self.rules is None or len(self.rules) == 0:
            return []
        
        # Find rules where all basket items are in antecedents
        basket_set = set(basket)
        basket_rules = self.rules[
            self.rules['antecedents'].apply(lambda x: basket_set.issubset(x))
        ].copy()
        
        if len(basket_rules) == 0:
            # Fallback: find rules where any basket item is in antecedents
            basket_rules = self.rules[
                self.rules['antecedents'].apply(lambda x: len(basket_set.intersection(x)) > 0)
            ].copy()
        
        if len(basket_rules) == 0:
            return []
        
        # Sort by confidence and lift
        basket_rules = basket_rules.sort_values(['confidence', 'lift'], ascending=False)
        
        recommendations = []
        seen_products = set(basket)
        
        for _, rule in basket_rules.iterrows():
            consequent = list(rule['consequents'])[0]
            if consequent not in seen_products:
                recommendations.append({
                    'product': consequent,
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support'],
                    'explanation': f"{rule['confidence']:.1%} of customers who bought {', '.join(basket)} also bought {consequent}"
                })
                seen_products.add(consequent)
                
                if len(recommendations) >= top_n:
                    break
        
        return recommendations

class CollaborativeFilteringRecommender:
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.user_item_matrix = None
        self.products = None
        
    def fit(self, user_item_matrix: pd.DataFrame) -> 'CollaborativeFilteringRecommender':
        """Fit the collaborative filtering model using SVD"""
        self.user_item_matrix = user_item_matrix
        self.products = list(user_item_matrix.columns)
        
        # Apply SVD
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(user_item_matrix)
        self.item_factors = self.svd.components_.T
        
        return self
    
    def get_user_recommendations(self, user_id: int, top_n: int = 5) -> List[Dict]:
        """Get recommendations for a specific user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's purchase history
        user_purchases = self.user_item_matrix.loc[user_id]
        purchased_products = user_purchases[user_purchases > 0].index.tolist()
        
        # Calculate predicted ratings for all products
        user_vector = self.user_factors[self.user_item_matrix.index.get_loc(user_id)]
        predicted_ratings = np.dot(self.item_factors, user_vector)
        
        # Create recommendations
        product_scores = list(zip(self.products, predicted_ratings))
        product_scores.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for product, score in product_scores:
            if product not in purchased_products:
                recommendations.append({
                    'product': product,
                    'score': score,
                    'explanation': f"Based on similar customers' preferences"
                })
                
                if len(recommendations) >= top_n:
                    break
        
        return recommendations
    
    def get_product_similarity(self, product: str, top_n: int = 5) -> List[Dict]:
        """Get products similar to a given product"""
        if product not in self.products:
            return []
        
        product_idx = self.products.index(product)
        product_vector = self.item_factors[product_idx]
        
        # Calculate similarities with all products
        similarities = []
        for i, other_product in enumerate(self.products):
            if other_product != product:
                other_vector = self.item_factors[i]
                similarity = cosine_similarity([product_vector], [other_vector])[0][0]
                similarities.append((other_product, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for other_product, similarity in similarities[:top_n]:
            recommendations.append({
                'product': other_product,
                'similarity': similarity,
                'explanation': f"Similar to {product} based on customer preferences"
            })
        
        return recommendations

class HybridRecommender:
    def __init__(self, mba_weight: float = 0.6, cf_weight: float = 0.4):
        self.mba_weight = mba_weight
        self.cf_weight = cf_weight
        self.mba = MarketBasketAnalyzer()
        self.cf = CollaborativeFilteringRecommender()
        
    def fit(self, baskets: List[List[str]], user_item_matrix: pd.DataFrame) -> 'HybridRecommender':
        """Fit both models"""
        self.mba.fit(baskets)
        self.cf.fit(user_item_matrix)
        return self
    
    def get_recommendations(self, product: str, top_n: int = 5) -> List[Dict]:
        """Get hybrid recommendations for a product"""
        # Get MBA recommendations
        mba_recs = self.mba.get_recommendations(product, top_n * 2)
        
        # Get CF recommendations
        cf_recs = self.cf.get_product_similarity(product, top_n * 2)
        
        # Combine and score
        combined_scores = {}
        
        # Add MBA scores
        for rec in mba_recs:
            product_name = rec['product']
            score = rec['confidence'] * self.mba_weight
            combined_scores[product_name] = {
                'product': product_name,
                'score': score,
                'mba_confidence': rec['confidence'],
                'explanation': rec['explanation']
            }
        
        # Add CF scores
        for rec in cf_recs:
            product_name = rec['product']
            cf_score = rec['similarity'] * self.cf_weight
            
            if product_name in combined_scores:
                combined_scores[product_name]['score'] += cf_score
                combined_scores[product_name]['cf_similarity'] = rec['similarity']
            else:
                combined_scores[product_name] = {
                    'product': product_name,
                    'score': cf_score,
                    'cf_similarity': rec['similarity'],
                    'explanation': rec['explanation']
                }
        
        # Sort by combined score
        final_recommendations = list(combined_scores.values())
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return final_recommendations[:top_n]

if __name__ == "__main__":
    # Test the recommenders
    from data_prep import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    baskets = preprocessor.get_basket_data()
    user_item_matrix = preprocessor.get_transaction_matrix()
    
    print("Testing Market Basket Analysis...")
    mba = MarketBasketAnalyzer()
    mba.fit(baskets)
    recs = mba.get_recommendations("White T-shirt", 3)
    print(f"MBA recommendations for White T-shirt: {recs}")
    
    print("\nTesting Collaborative Filtering...")
    cf = CollaborativeFilteringRecommender()
    cf.fit(user_item_matrix)
    recs = cf.get_product_similarity("White T-shirt", 3)
    print(f"CF recommendations for White T-shirt: {recs}")
