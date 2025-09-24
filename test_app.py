#!/usr/bin/env python3
"""
Test script for StyleHive Dashboard
Run this to verify the app works correctly before deployment
"""

import sys
import os
import traceback

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.decomposition import TruncatedSVD
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
        import networkx as nx
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_generation():
    """Test data generation"""
    print("🔍 Testing data generation...")
    try:
        sys.path.append('utils')
        from data_generator import StyleHiveDataGenerator
        
        generator = StyleHiveDataGenerator()
        df = generator.generate_dataset(months=1)  # Test with 1 month
        
        if len(df) > 0:
            print(f"✅ Data generation successful: {len(df)} transactions")
            return True
        else:
            print("❌ No data generated")
            return False
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing"""
    print("🔍 Testing data preprocessing...")
    try:
        sys.path.append('utils')
        from data_prep import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data()
        
        if len(df) > 0:
            print(f"✅ Data preprocessing successful: {df.shape[0]} rows")
            return True
        else:
            print("❌ No data loaded")
            return False
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False

def test_models():
    """Test ML models"""
    print("🔍 Testing ML models...")
    try:
        sys.path.append('utils')
        from data_prep import DataPreprocessor
        from recommenders import MarketBasketAnalyzer, CollaborativeFilteringRecommender
        from insights import BusinessInsights
        
        # Load data
        preprocessor = DataPreprocessor()
        preprocessor.load_data()
        baskets = preprocessor.get_basket_data()
        user_item_matrix = preprocessor.get_transaction_matrix()
        
        # Test Market Basket Analysis
        mba = MarketBasketAnalyzer(min_support=0.05, min_confidence=0.3)
        mba.fit(baskets)
        print("✅ Market Basket Analysis model created")
        
        # Test Collaborative Filtering
        cf = CollaborativeFilteringRecommender(n_components=3)
        cf.fit(user_item_matrix)
        print("✅ Collaborative Filtering model created")
        
        # Test Insights
        insights = BusinessInsights(preprocessor)
        kpis = insights.get_kpis()
        print(f"✅ Business insights generated: {len(kpis)} KPIs")
        
        return True
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        traceback.print_exc()
        return False

def test_app_components():
    """Test app components"""
    print("🔍 Testing app components...")
    try:
        sys.path.append('.')
        from app import load_data, load_models, create_top_products_chart
        from insights import BusinessInsights
        
        # Test data loading
        preprocessor = load_data()
        print("✅ Data loading function works")
        
        # Test model loading
        mba, cf, hybrid = load_models()
        print("✅ Model loading functions work")
        
        # Test insights
        insights = BusinessInsights(preprocessor)
        top_products = insights.get_top_products()
        print("✅ Insights generation works")
        
        # Test chart creation
        fig = create_top_products_chart(top_products)
        print("✅ Chart creation works")
        
        return True
    except Exception as e:
        print(f"❌ App component testing failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test file structure"""
    print("🔍 Testing file structure...")
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'utils/data_generator.py',
        'utils/data_prep.py',
        'utils/recommenders.py',
        'utils/insights.py',
        'sample_data/stylehive_transactions.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all tests"""
    print("🧪 Running StyleHive Dashboard Tests")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Data Preprocessing", test_data_preprocessing),
        ("ML Models", test_models),
        ("App Components", test_app_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
