# StyleHive Fashion Recommendation Dashboard

[![CI/CD Pipeline](https://github.com/heritai/stylehive-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/heritai/stylehive-dashboard/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stylehive.streamlit.app/)

A comprehensive analytics and recommendation platform designed for **StyleHive**, a fictional online fashion retailer. This project demonstrates how data science and machine learning can drive revenue growth through intelligent product recommendations and actionable business insights.

## 🎯 Business Context

**Problem**: StyleHive struggles with low average order values, as customers typically purchase only 1-2 items per transaction, missing significant cross-selling opportunities.

**Solution**: Powered by Market Basket Analysis and Collaborative Filtering, this dynamic recommendation dashboard suggests complementary products, designed to increase basket size and enhance customer satisfaction.

## ✨ Features

### 📊 Global Insights Dashboard
-   **Key Performance Indicators (KPIs)**: Track total transactions, unique customers, average basket size, and key revenue metrics.
-   **Top-Selling Products**: Visualize product performance with interactive bar charts.
-   **Product Co-occurrence Heatmap**: A visual representation of frequently co-purchased items.
-   **Product Affinity Network**: Explore product relationships and associations through a dynamic network graph.

### 🔍 Recommendation Explorer
-   **Product-Specific Recommendations**: Get complementary item suggestions by selecting any product.
-   **Multiple Recommendation Engines**:
    -   **"People Who Bought X Also Bought Y" Insights**: Utilizes Market Basket Analysis.
    -   **Collaborative Filtering**: Suggests items based on similar customer preferences.
    -   **Hybrid Approach**: Offers optimized recommendations by combining MBA and Collaborative Filtering for superior results.
-   **Confidence Scores**: Each recommendation includes clear confidence levels and explanations.

### 🛒 Customer Basket Simulation
-   **Interactive Basket Builder**: Simulate customer shopping carts by selecting multiple products.
-   **Smart Recommendations**: Receive intelligent suggestions for additional items based on the current basket.
-   **Basket Analysis**: Gain insights into the strength of product combinations within the simulated basket.

### 💡 Business Intelligence
-   **Strategic Recommendations**: Generate data-driven insights to inform business strategy.
-   **Customer Segmentation**: Analyze customers across high, medium, and low-value segments.
-   **Seasonal Patterns**: Uncover seasonal purchasing behaviors and trends.
-   **Co-purchase Insights**: Identify key product combinations that drive sales.

## 🛠️ Tech Stack

-   **Python 3.10+**
-   **Streamlit**: Interactive web dashboard development.
-   **scikit-learn**: Machine learning models and SVD.
-   **mlxtend**: Market basket analysis (Apriori algorithm).
-   **pandas & numpy**: Data manipulation and analysis.
-   **plotly**: Interactive visualizations.
-   **networkx**: Network graph analysis.

## 📊 Dataset

This project leverages a **synthetic, yet realistic dataset** comprising 18 months of transaction data for 10 distinct fashion products:

1.  White T-shirt
2.  Blue Jeans
3.  Sneakers
4.  Leather Jacket
5.  Sunglasses
6.  Backpack
7.  Hoodie
8.  Formal Shirt
9.  Dress Shoes
10. Smartwatch

### Key Realistic Patterns:
-   **Co-purchases**: E.g., T-shirt + Jeans + Sneakers, Jacket + Sunglasses, Formal Shirt + Dress Shoes.
-   **Cross-sell Opportunities**: E.g., Smartwatch with Sneakers, Sunglasses with Jackets.
-   **Seasonal Trends**: Reflecting summer items in warm months and outerwear in winter.
-   **Customer Behavior**: Varied purchase frequencies and basket sizes.

## 🚀 Quick Start

### Prerequisites
-   Python 3.10+ and pip package manager.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd stylehive-recommender-dashboard
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the dashboard**:
    ```bash
    streamlit run app.py
    ```

4.  **Access the dashboard** in your browser at `http://localhost:8501`.

### Testing

To ensure all components function as expected, run the comprehensive test suite:

```bash
python test_app.py
```

This will test:
-   ✅ All required imports
-   ✅ Data generation and preprocessing
-   ✅ ML model creation and training
-   ✅ Streamlit app component functionality
-   ✅ File structure and dependency validation

## 📁 Project Structure

```
stylehive-recommender-dashboard/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── utils/                          # Utility modules
│   ├── data_generator.py           # Synthetic data generation
│   ├── data_prep.py                # Data preprocessing utilities
│   ├── recommenders.py             # ML recommendation models
│   └── insights.py                 # Business insights generation
├── sample_data/                    # Generated datasets
│   └── stylehive_transactions.csv
└── reports/                        # Generated reports (optional)
    └── example_report.pdf
```

## 🧠 Machine Learning Models

### Market Basket Analysis
-   **Algorithm**: Apriori algorithm for frequent itemset mining.
-   **Purpose**: Identifies products frequently purchased together.
-   **Output**: Association rules, including confidence and lift metrics.
-   **Use Case**: Powers "Customers who bought X also bought Y" recommendations.

### Collaborative Filtering
-   **Algorithm**: Singular Value Decomposition (SVD).
-   **Purpose**: Discovers similar customers and products based on historical preferences.
-   **Output**: Product similarity scores derived from customer interactions.
-   **Use Case**: Enables "Customers like you also bought..." recommendations.

### Hybrid Recommendation System
-   **Approach**: Synergistically combines Market Basket Analysis (MBA) and Collaborative Filtering (CF).
-   **Weighting**: Utilizes configurable weights for each underlying model.
-   **Output**: Delivers optimized recommendations by integrating multiple signals.

## 📈 Business Impact

This dashboard demonstrates how fashion retailers can:

1.  **Increase Average Order Value**: Effectively boost Average Order Value (AOV) by suggesting complementary products.
2.  **Enhance Customer Experience**: Deliver a personalized shopping journey through tailored recommendations.
3.  **Optimize Inventory Management**: Improve inventory planning by uncovering key product relationships.
4.  **Inform Strategic Decisions**: Provide data-driven insights for robust business strategy development.

## 🔧 Customization

### Extend Product Catalog
1.  Update the product list in `utils/data_generator.py`.
2.  Define co-purchase patterns in the `co_purchase_patterns` dictionary.
3.  Regenerate the dataset.

### Fine-tune Recommendation Models
1.  Modify confidence thresholds in `MarketBasketAnalyzer`.
2.  Change SVD components in `CollaborativeFilteringRecommender`.
3.  Adjust hybrid weights in `HybridRecommender`.

### Integrate New Visualizations
1.  Extend the dashboard in `app.py`.
2.  Add new chart types using Plotly.
3.  Create custom business metrics in `insights.py`.

## ⚠️ Important Notes

-   **Synthetic Data**: This project uses synthetically generated data for demonstration purposes only.
-   **Simplified Models**: While effective for demonstration, real-world systems typically require more sophisticated algorithms and models.
-   **Scalability**: Production-grade systems require further considerations for scalability and performance with large datasets.
-   **Privacy**: Handling real customer data mandates stringent privacy and security protocols.

## 🧪 CI/CD Pipeline

This project includes automated testing and deployment via GitHub Actions:

### GitHub Actions Workflow
-   **Automated Testing**: Executes automatically on every push and pull request.
-   **Data Generation Tests**: Validates synthetic data generation.
-   **Model Training Tests**: Confirms correct functionality of ML model training.
-   **App Component Tests**: Verifies Streamlit application components.
-   **Dependency Validation**: Ensures all required dependencies are met.

### Local Testing
```bash
# Run comprehensive test suite
python test_app.py

# Test specific components (example)
python -c "from app import load_data; print('✅ App loads data successfully')"
```

### Continuous Integration Benefits
-   ✅ **Early Bug Detection**: Identifies issues proactively, preventing deployment of faulty code.
-   ✅ **Automated Validation**: Streamlines development by eliminating manual testing.
-   ✅ **Deployment Confidence**: Ensures only validated code is deployed.
-   ✅ **Practical Usage Examples**: Test results serve as practical usage examples.

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
1.  **Push to GitHub:**
    ```bash
    git init
    git add .
    git commit -m "Initial commit: StyleHive Fashion Dashboard"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/stylehive-dashboard.git
    git push -u origin main
    ```

2.  **Deploy on Streamlit Cloud:**
    -   Go to [share.streamlit.io](https://share.streamlit.io)
    -   Click "New app"
    -   Connect your GitHub account
    -   Select your repository: `YOUR_USERNAME/stylehive-dashboard`
    -   Set main file path: `app.py`
    -   Click "Deploy"

3.  **Your application will be live at:**
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

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stylehive.streamlit.app)

*Note: The badge above links to a live demo.*

## 🤝 Contributing

This is a demonstration project, but contributions are welcome:

1.  Fork the repository.
2.  Create a feature branch.
3.  Make your changes.
4.  Submit a pull request.

## 📄 License

This project is for educational and demonstration purposes. Please ensure compliance with any applicable licenses for the libraries used.

## 📞 Contact

For questions about this project or fashion retail analytics, please open an issue in the repository.

---

**StyleHive Dashboard** — *Empowering Fashion Retail with Data-Driven Intelligence* 👗📊