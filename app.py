"""
StyleHive Fashion Recommendation Dashboard
A comprehensive dashboard for fashion retail insights and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_prep import DataPreprocessor
from recommenders import MarketBasketAnalyzer, CollaborativeFilteringRecommender, HybridRecommender
from insights import BusinessInsights

# Page configuration
st.set_page_config(
    page_title="StyleHive Dashboard",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* CLEAN, SIMPLE STYLING */
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    /* TAB STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E86AB;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB;
        color: white;
        box-shadow: 0 4px 8px rgba(46, 134, 171, 0.3);
        border-color: #2E86AB;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8f4f8;
        color: #2E86AB;
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(46, 134, 171, 0.2);
        border-color: #2E86AB;
    }
    
    /* CARD STYLES */
    .metric-card {
        background-color: #f0f2f6;
        color: #333;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    
    .recommendation-card {
        background-color: #e8f4f8;
        color: #333;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #A23B72;
    }
    
    .insight-box {
        background-color: #f8f9fa;
        color: #333;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    /* HIDE SIDEBAR */
    .css-1d391kg {
        display: none;
    }
    
    /* DARK MODE SUPPORT */
    @media (prefers-color-scheme: dark) {
        .subtitle {
            color: #4a9eff;
        }
        
        .metric-card {
            background-color: #2d3748;
            color: #e2e8f0;
            border-left-color: #4a9eff;
        }
        
        .recommendation-card {
            background-color: #2d3748;
            color: #e2e8f0;
            border-left-color: #ff6b9d;
        }
        
        .insight-box {
            background-color: #2d3748;
            color: #e2e8f0;
            border-color: #4a5568;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #e0e0e0;
            background-color: #2d3748;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #4a9eff;
            color: white;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #4a5568;
            color: #e0e0e0;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load and cache data"""
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    return preprocessor

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_models():
    """Load and cache ML models"""
    # Create a new preprocessor instance for caching
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    baskets = preprocessor.get_basket_data()
    user_item_matrix = preprocessor.get_transaction_matrix()
    
    # Initialize models with optimized parameters
    mba = MarketBasketAnalyzer(min_support=0.02, min_confidence=0.2)  # Higher thresholds for speed
    cf = CollaborativeFilteringRecommender(n_components=5)  # Fewer components for speed
    hybrid = HybridRecommender()
    
    # Fit models
    mba.fit(baskets)
    cf.fit(user_item_matrix)
    hybrid.fit(baskets, user_item_matrix)
    
    return mba, cf, hybrid

def create_kpi_metrics(kpis):
    """Create KPI metrics display"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Transactions",
            value=f"{kpis['total_transactions']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Unique Customers",
            value=f"{kpis['unique_customers']:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Avg Basket Size",
            value=f"{kpis['avg_basket_size']:.1f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Avg Order Value",
            value=f"${kpis['avg_order_value']:.0f}",
            delta=None
        )

def create_top_products_chart(top_products):
    """Create top products bar chart"""
    fig = px.bar(
        top_products,
        x='TotalPurchases',
        y='Product',
        orientation='h',
        title="Top Selling Products",
        color='TotalPurchases',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        height=400,
        yaxis=dict(
            categoryorder='total ascending',
            gridcolor='rgba(128,128,128,0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)')
    )
    return fig

def create_co_occurrence_heatmap(co_occurrence):
    """Create product co-occurrence heatmap"""
    fig = px.imshow(
        co_occurrence.values,
        x=co_occurrence.columns,
        y=co_occurrence.index,
        color_continuous_scale='Blues',
        title="Product Co-occurrence Heatmap"
    )
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
    )
    return fig

def create_network_graph(network_data):
    """Create product affinity network graph"""
    G = nx.Graph()
    
    # Add nodes
    for node in network_data['nodes']:
        G.add_node(node)
    
    # Add edges
    for edge in network_data['edges'][:15]:  # Top 15 connections
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    # Calculate layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=20,
            color='#2E86AB',
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Product Affinity Network',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Products frequently bought together",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="gray", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=500,
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)',
                       font=dict(color='#333333')
                   ))
    
    return fig

def main():
    """Main dashboard application"""
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
    
    # Header
    st.markdown('<h1 class="main-header">👗 StyleHive Fashion Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Fashion Recommendation & Business Insights Platform</p>', unsafe_allow_html=True)
    
    # Load data only once
    if not st.session_state.data_loaded:
        with st.spinner("Loading data and models..."):
            preprocessor = load_data()
            mba, cf, hybrid = load_models()
            insights = BusinessInsights(preprocessor)
            
            # Store in session state
            st.session_state.preprocessor = preprocessor
            st.session_state.mba = mba
            st.session_state.cf = cf
            st.session_state.hybrid = hybrid
            st.session_state.insights = insights
            st.session_state.data_loaded = True
    else:
        # Use cached data
        preprocessor = st.session_state.preprocessor
        mba = st.session_state.mba
        cf = st.session_state.cf
        hybrid = st.session_state.hybrid
        insights = st.session_state.insights
    
    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs(["📊 Global Insights", "🔍 Recommendation Explorer", "🛒 Basket Simulation"])
    
    with tab1:
        st.header("📊 Global Business Insights")
        
        # KPIs
        st.subheader("Key Performance Indicators")
        kpis = insights.get_kpis()
        create_kpi_metrics(kpis)
        
        # Business insight from KPIs
        st.markdown("""
        <div class="insight-box">
            <strong>💡 Business Insight:</strong> Your average basket size of {:.1f} items and order value of ${:.0f} 
            suggests opportunities for cross-selling. Consider promoting complementary products to increase revenue per transaction.
        </div>
        """.format(kpis['avg_basket_size'], kpis['avg_order_value']), unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Selling Products")
            top_products = insights.get_top_products()
            fig_products = create_top_products_chart(top_products)
            st.plotly_chart(fig_products, use_container_width=True)
            
            # Business insight for top products
            top_product = top_products.iloc[0]['Product']
            top_purchases = top_products.iloc[0]['TotalPurchases']
            st.markdown(f"""
            <div class="insight-box">
                <strong>🎯 Strategic Focus:</strong> {top_product} is your best seller with {top_purchases} purchases. 
                Use this as a foundation for cross-selling opportunities and bundle promotions.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Product Co-occurrence")
            co_occurrence = preprocessor.get_co_occurrence_matrix()
            fig_heatmap = create_co_occurrence_heatmap(co_occurrence)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Business insight for co-occurrence
            st.markdown("""
            <div class="insight-box">
                <strong>🔗 Cross-Sell Opportunities:</strong> The heatmap shows which products are frequently bought together. 
                Focus on promoting these combinations to increase average order value.
            </div>
            """, unsafe_allow_html=True)
        
        # Network graph
        st.subheader("Product Affinity Network")
        network_data = insights.get_product_affinity_network()
        fig_network = create_network_graph(network_data)
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Co-purchase insights with business recommendations
        st.subheader("Key Co-purchase Insights & Recommendations")
        co_purchase_insights = insights.get_co_purchase_insights()
        
        col1, col2 = st.columns(2)
        with col1:
            for i, insight in enumerate(co_purchase_insights[:3]):
                st.markdown(f"""
                <div class="insight-box">
                    <strong>{insight['product1']} + {insight['product2']}</strong><br>
                    {insight['percentage']}% co-purchase rate ({insight['co_purchases']} transactions)<br>
                    <em>💡 Recommendation: Bundle these products or show {insight['product2']} when customers view {insight['product1']}</em>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            for i, insight in enumerate(co_purchase_insights[3:6]):
                st.markdown(f"""
                <div class="insight-box">
                    <strong>{insight['product1']} + {insight['product2']}</strong><br>
                    {insight['percentage']}% co-purchase rate ({insight['co_purchases']} transactions)<br>
                    <em>💡 Recommendation: Create "Complete the Look" suggestions for these combinations</em>
                </div>
                """, unsafe_allow_html=True)
        
        # Customer segmentation insights
        st.subheader("Customer Behavior Analysis")
        customer_segments = insights.get_customer_segments()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Customer Segment Distribution:**")
            segment_df = pd.DataFrame(
                list(customer_segments['segment_distribution'].items()),
                columns=['Segment', 'Count']
            )
            fig_segments = px.pie(
                segment_df,
                values='Count',
                names='Segment',
                title="Customer Segments"
            )
            fig_segments.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333333')
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with col2:
            st.write("**Segment Characteristics:**")
            segment_stats = customer_segments['customer_stats'].groupby('Segment').agg({
                'TotalPurchases': 'mean',
                'ActiveDays': 'mean'
            }).round(1)
            st.dataframe(segment_stats)
            
            # Business insight for customer segments
            high_value_pct = (customer_segments['segment_distribution'].get('High Value', 0) / 
                             sum(customer_segments['segment_distribution'].values())) * 100
            st.markdown(f"""
            <div class="insight-box">
                <strong>👥 Customer Strategy:</strong> {high_value_pct:.1f}% of your customers are high-value. 
                Focus retention efforts on this segment while working to upgrade medium-value customers.
            </div>
            """, unsafe_allow_html=True)
        
        # Seasonal insights
        st.subheader("Seasonal Purchasing Patterns")
        seasonal_insights = insights.get_seasonal_insights()
        
        st.write("**Seasonal Champions (Top Product per Season):**")
        for season, champion in seasonal_insights['seasonal_champions'].items():
            st.write(f"• **{season}**: {champion['product']} ({champion['purchases']} purchases)")
        
        # Seasonal trends chart
        seasonal_trends = seasonal_insights['seasonal_trends']
        if seasonal_trends:
            trend_data = []
            for product, seasons in seasonal_trends.items():
                for season, purchases in seasons.items():
                    trend_data.append({
                        'Product': product,
                        'Season': season,
                        'Purchases': purchases
                    })
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                fig_trends = px.bar(
                    trend_df,
                    x='Season',
                    y='Purchases',
                    color='Product',
                    title="Seasonal Purchase Patterns",
                    barmode='group'
                )
                fig_trends.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#333333')
                )
                st.plotly_chart(fig_trends, use_container_width=True)
                
                # Seasonal business insight
                st.markdown("""
                <div class="insight-box">
                    <strong>📅 Seasonal Strategy:</strong> Use seasonal patterns to optimize inventory and marketing. 
                    Promote summer items in warm months and outerwear in winter. Plan seasonal campaigns around peak products.
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.header("🔍 Product Recommendation Explorer")
        
        # Product selection
        products = sorted(preprocessor.df['Product'].unique())
        selected_product = st.selectbox(
            "Select a product to get recommendations:",
            products,
            key="product_selector"
        )
        
        if selected_product:
            # Product performance insight
            product_stats = preprocessor.get_product_statistics()
            product_info = product_stats[product_stats['Product'] == selected_product].iloc[0]
            
            st.markdown(f"""
            <div class="insight-box">
                <strong>📊 Product Performance:</strong> {selected_product} has been purchased {int(product_info['TotalPurchases'])} times 
                by {int(product_info['UniqueCustomers'])} unique customers, with an average of {product_info['AvgPurchasesPerCustomer']:.1f} 
                purchases per customer.
            </div>
            """, unsafe_allow_html=True)
            
            # Get recommendations from different models
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🛒 Market Basket Analysis")
                mba_recs = mba.get_recommendations(selected_product, 5)
                if mba_recs:
                    st.markdown("**💡 Business Insight:** These recommendations are based on actual purchase patterns - customers who bought this item also bought these products.")
                    for rec in mba_recs:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <strong>{rec['product']}</strong><br>
                            Confidence: {rec['confidence']:.1%}<br>
                            <small>{rec['explanation']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No strong associations found for this product.")
            
            with col2:
                st.subheader("👥 Collaborative Filtering")
                cf_recs = cf.get_product_similarity(selected_product, 5)
                if cf_recs:
                    st.markdown("**💡 Business Insight:** These recommendations are based on customer behavior patterns - products that appeal to similar customer segments.")
                    for rec in cf_recs:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <strong>{rec['product']}</strong><br>
                            Similarity: {rec['similarity']:.2f}<br>
                            <small>{rec['explanation']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No similar products found.")
            
            with col3:
                st.subheader("🎯 Hybrid Recommendations")
                hybrid_recs = hybrid.get_recommendations(selected_product, 5)
                if hybrid_recs:
                    st.markdown("**💡 Business Insight:** These are the optimal recommendations combining both purchase patterns and customer preferences for maximum effectiveness.")
                    for rec in hybrid_recs:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <strong>{rec['product']}</strong><br>
                            Score: {rec['score']:.3f}<br>
                            <small>Combined MBA + CF approach</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recommendations available.")
            
            # Implementation recommendations
            st.subheader("🚀 Implementation Strategy")
            if mba_recs or cf_recs or hybrid_recs:
                st.markdown("""
                <div class="insight-box">
                    <strong>💼 Action Items:</strong>
                    <ul>
                        <li><strong>Product Pages:</strong> Show these recommendations on the {selected_product} product page</li>
                        <li><strong>Email Marketing:</strong> Include these items in abandoned cart and follow-up emails</li>
                        <li><strong>Bundle Offers:</strong> Create "Complete the Look" bundles with the top recommendations</li>
                        <li><strong>Checkout Upsells:</strong> Display these items during the checkout process</li>
                    </ul>
                </div>
                """.format(selected_product=selected_product), unsafe_allow_html=True)
    
    with tab3:
        st.header("🛒 Customer Basket Simulation")
        
        st.markdown("**Simulate a customer's shopping basket and get recommendations for additional items.**")
        
        # Multi-select for basket
        available_products = sorted(preprocessor.df['Product'].unique())
        selected_basket = st.multiselect(
            "Select products for your basket:",
            available_products,
            default=[],
            key="basket_products"
        )
        
        if selected_basket:
            st.subheader(f"Your Basket ({len(selected_basket)} items)")
            for product in selected_basket:
                st.write(f"• {product}")
            
            # Basket analysis with insights
            st.subheader("📊 Basket Analysis & Insights")
            basket_products = set(selected_basket)
            co_occurrence = preprocessor.get_co_occurrence_matrix()
            
            # Find strongest co-purchases within the basket
            basket_strength = 0
            for i, product1 in enumerate(selected_basket):
                for j, product2 in enumerate(selected_basket):
                    if i != j:
                        strength = co_occurrence.loc[product1, product2]
                        basket_strength += strength
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Basket Co-purchase Strength", f"{basket_strength:.0f}")
            
            with col2:
                # Calculate estimated basket value
                price_mapping = {
                    'White T-shirt': 25, 'Blue Jeans': 60, 'Sneakers': 80, 'Leather Jacket': 200,
                    'Sunglasses': 50, 'Backpack': 40, 'Hoodie': 45, 'Formal Shirt': 70,
                    'Dress Shoes': 120, 'Smartwatch': 300
                }
                basket_value = sum(price_mapping.get(product, 0) for product in selected_basket)
                st.metric("Estimated Basket Value", f"${basket_value}")
            
            with col3:
                # Calculate potential upsell value
                if basket_value > 0:
                    potential_upsell = basket_value * 0.3  # 30% potential increase
                    st.metric("Upsell Potential", f"${potential_upsell:.0f}")
            
            # Basket strength insights
            if basket_strength > 10:
                st.success("✅ Strong basket combination - these items are frequently bought together!")
                st.markdown("""
                <div class="insight-box">
                    <strong>💡 Insight:</strong> This is a well-balanced basket that customers typically purchase together. 
                    Consider creating a "Complete the Look" bundle for this combination.
                </div>
                """, unsafe_allow_html=True)
            elif basket_strength > 5:
                st.warning("⚠️ Moderate basket combination")
                st.markdown("""
                <div class="insight-box">
                    <strong>💡 Insight:</strong> This basket has some complementary items but could benefit from additional accessories 
                    or related products to increase value.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Unusual basket combination - consider adding complementary items")
                st.markdown("""
                <div class="insight-box">
                    <strong>💡 Insight:</strong> This is an uncommon combination. Consider suggesting complementary items 
                    to create a more cohesive look and increase customer satisfaction.
                </div>
                """, unsafe_allow_html=True)
            
            # Get basket recommendations with caching
            st.subheader("🎯 Recommended Additional Items")
            
            # Create a cache key based on the basket
            basket_key = tuple(sorted(selected_basket))
            
            # Check if we have cached recommendations for this basket
            if 'basket_recommendations' not in st.session_state:
                st.session_state.basket_recommendations = {}
            
            if basket_key not in st.session_state.basket_recommendations:
                with st.spinner("Computing recommendations..."):
                    basket_recs = mba.get_basket_recommendations(selected_basket, 5)
                    st.session_state.basket_recommendations[basket_key] = basket_recs
            else:
                basket_recs = st.session_state.basket_recommendations[basket_key]
            
            if basket_recs:
                st.markdown("""
                <div class="insight-box">
                    <strong>💼 Business Strategy:</strong> These recommendations are specifically tailored to your current basket. 
                    They're based on what other customers typically add to similar purchases.
                </div>
                """, unsafe_allow_html=True)
                
                for rec in basket_recs:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>{rec['product']}</strong><br>
                        Confidence: {rec['confidence']:.1%}<br>
                        <small>{rec['explanation']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Implementation strategy for basket
                st.subheader("🚀 Implementation Strategy")
                st.markdown("""
                <div class="insight-box">
                    <strong>💼 Action Items for This Basket:</strong>
                    <ul>
                        <li><strong>Checkout Page:</strong> Display these recommendations as "You might also like" during checkout</li>
                        <li><strong>Bundle Creation:</strong> Create a "Complete the Look" bundle with your current items + top recommendations</li>
                        <li><strong>Email Follow-up:</strong> Send a follow-up email with these suggestions after purchase</li>
                        <li><strong>Retargeting Ads:</strong> Use these items in retargeting campaigns for similar customer segments</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No additional recommendations for this basket combination.")
                st.markdown("""
                <div class="insight-box">
                    <strong>💡 Insight:</strong> This basket combination is unique. Consider analyzing customer feedback 
                    and creating new product bundles based on this pattern.
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>StyleHive Fashion Recommendation Dashboard | Built with Streamlit & Python</p>
        <p><em>This is a demonstration project with synthetic data for educational purposes.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
