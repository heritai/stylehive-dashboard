# StyleHive Fashion Recommendation Dashboard

[![CI/CD Pipeline](https://github.com/heritai/stylehive-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/heritai/stylehive-dashboard/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stylehive-dashboard.streamlit.app/)

A comprehensive fashion retail analytics and recommendation platform built for **StyleHive**, a fictive online fashion retailer. This project demonstrates how data science and machine learning can be applied to boost revenue through intelligent product recommendations and business insights.

## 🎯 Business Context

**Problem**: StyleHive customers typically purchase only 1-2 items per transaction, resulting in lower average order values and missed cross-selling opportunities.

**Solution**: A recommendation dashboard that suggests complementary products to increase basket size and customer satisfaction, powered by market basket analysis and collaborative filtering.

## ✨ Features

### 📊 Global Insights Dashboard
- **Key Performance Indicators**: Total transactions, unique customers, average basket size, revenue metrics
- **Top Selling Products**: Interactive bar charts showing product performance
- **Product Co-occurrence Heatmap**: Visual representation of which products are frequently bought together
- **Product Affinity Network**: Network graph showing product relationships and associations

### 🔍 Recommendation Explorer
- **Product-Specific Recommendations**: Select any product to see complementary items
- **Multiple Recommendation Engines**:
  - **Market Basket Analysis**: "People who bought X also bought Y" insights
  - **Collaborative Filtering**: Similar customer preferences
  - **Hybrid Approach**: Combined MBA + CF for optimal recommendations
- **Confidence Scores**: Each recommendation includes confidence levels and explanations

### 🛒 Customer Basket Simulation
- **Interactive Basket Builder**: Select multiple products to simulate a customer's cart
- **Smart Recommendations**: Get suggestions for additional items based on current basket
- **Basket Analysis**: Understand the strength of product combinations in your basket

### 💡 Business Intelligence
- **Strategic Recommendations**: Data-driven insights for business strategy
- **Customer Segmentation**: High-value, medium-value, and low-value customer analysis
- **Seasonal Patterns**: Understanding seasonal purchasing behaviors
- **Co-purchase Insights**: Key product combinations that drive sales

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit**: Interactive web dashboard
- **scikit-learn**: Machine learning models and SVD
- **mlxtend**: Market basket analysis (Apriori algorithm)
- **pandas & numpy**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **networkx**: Network graph analysis

## 📊 Dataset

The project uses a **synthetic but realistic dataset** containing 18 months of transaction data for 10 fashion products:

1. White T-shirt
2. Blue Jeans
3. Sneakers
4. Leather Jacket
5. Sunglasses
6. Backpack
7. Hoodie
8. Formal Shirt
9. Dress Shoes
10. Smartwatch

### Realistic Patterns Included:
- **Co-purchases**: T-shirt + Jeans + Sneakers, Jacket + Sunglasses, Formal Shirt + Dress Shoes
- **Cross-sell opportunities**: Smartwatch with Sneakers, Sunglasses with Jackets
- **Seasonal trends**: Summer items in warm months, outerwear in winter
- **Customer behavior**: Different purchase frequencies and basket sizes

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd stylehive-recommender-dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Testing

Run the test suite to verify everything works correctly:

```bash
python test_app.py
```

This will test:
- ✅ All required imports
- ✅ Data generation and preprocessing
- ✅ ML model creation and training
- ✅ App component functionality
- ✅ File structure and dependencies

## 📁 Project Structure

```
stylehive-recommender-dashboard/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── utils/                         # Utility modules
│   ├── data_generator.py         # Synthetic data generation
│   ├── data_prep.py              # Data preprocessing utilities
│   ├── recommenders.py           # ML recommendation models
│   └── insights.py               # Business insights generation
├── sample_data/                  # Generated datasets
│   └── stylehive_transactions.csv
└── reports/                      # Generated reports (optional)
    └── example_report.pdf
```

## 🧠 Machine Learning Models

### Market Basket Analysis
- **Algorithm**: Apriori algorithm for frequent itemset mining
- **Purpose**: Find products frequently bought together
- **Output**: Association rules with confidence and lift metrics
- **Use Case**: "Customers who bought X also bought Y"

### Collaborative Filtering
- **Algorithm**: Singular Value Decomposition (SVD)
- **Purpose**: Find similar customers and products
- **Output**: Product similarity scores based on customer preferences
- **Use Case**: "Customers like you also bought..."

### Hybrid Recommendation System
- **Approach**: Combines Market Basket Analysis and Collaborative Filtering
- **Weighting**: Configurable weights for different models
- **Output**: Optimized recommendations with multiple signals

## 📈 Business Impact

This dashboard demonstrates how fashion retailers can:

1. **Increase Average Order Value**: By suggesting complementary products
2. **Improve Customer Experience**: Through personalized recommendations
3. **Optimize Inventory**: By understanding product relationships
4. **Drive Strategic Decisions**: With data-driven insights

## 🔧 Customization

### Adding New Products
1. Update the product list in `utils/data_generator.py`
2. Add co-purchase patterns in the `co_purchase_patterns` dictionary
3. Regenerate the dataset by running the data generator

### Adjusting Recommendation Models
- Modify confidence thresholds in `MarketBasketAnalyzer`
- Change SVD components in `CollaborativeFilteringRecommender`
- Adjust hybrid weights in `HybridRecommender`

### Adding New Visualizations
- Extend the dashboard in `app.py`
- Add new chart types using Plotly
- Create custom business metrics in `insights.py`

## ⚠️ Important Notes

- **Synthetic Data**: This project uses generated data for demonstration purposes
- **Simplified Models**: Real-world systems may require more sophisticated algorithms
- **Scalability**: Production systems need additional considerations for large datasets
- **Privacy**: Real customer data requires proper privacy and security measures

## 🧪 CI/CD Pipeline

This project includes automated testing and deployment:

### GitHub Actions Workflow
- **Automated Testing**: Runs on every push and pull request
- **Data Generation Tests**: Verifies synthetic data creation
- **Model Training Tests**: Ensures ML models work correctly
- **App Component Tests**: Validates Streamlit app functionality
- **Import Validation**: Checks all dependencies are available

### Local Testing
```bash
# Run comprehensive test suite
python test_app.py

# Test specific components
python -c "from app import load_data; print('✅ App loads successfully')"
```

### Continuous Integration Benefits
- ✅ **Early Bug Detection**: Catch issues before deployment
- ✅ **Automated Validation**: No manual testing required
- ✅ Deployment Confidence**: Only tested code gets deployed
- ✅ **Documentation**: Test results provide usage examples

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: StyleHive Fashion Dashboard"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/stylehive-dashboard.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select your repository: `YOUR_USERNAME/stylehive-dashboard`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Your app will be live at:**
   ```
   https://YOUR_APP_NAME.streamlit.app
   ```

### Docker Deployment
```dockerfile
FROM python:3.10-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📊 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP_NAME.streamlit.app)

*Replace with your actual Streamlit Cloud URL after deployment*

## 🤝 Contributing

This is a demonstration project, but contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and demonstration purposes. Please ensure compliance with any applicable licenses for the libraries used.

## 📞 Contact

For questions about this project or fashion retail analytics, please open an issue in the repository.

---

**StyleHive Dashboard** - *Empowering Fashion Retail with Data Science* 👗📊
