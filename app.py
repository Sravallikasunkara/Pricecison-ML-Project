import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from price_predictor import PricePredictor
from datetime import datetime
import base64
import hashlib

# Page config
st.set_page_config(
    page_title="AI Price Optimization System",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = PricePredictor()
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'users' not in st.session_state:
    st.session_state.users = {'admin': hashlib.sha256('admin123'.encode()).hexdigest()}
# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False
# if 'current_user' not in st.session_state:
#     st.session_state.current_user = None
# if 'auth_page' not in st.session_state:
#     st.session_state.auth_page = None

def display_status_tags(row):
    """Helper function to display status tags"""
    tags = []
    if 'stock_status' in row and row['stock_status'] == '⚠ Low Stock':
        tags.append("Low Stock")
    if 'expiry_status' in row and row['expiry_status'] == '⏳ Soon Expiring':
        tags.append("Soon Expiring")
    if 'promotion_active' in row and row['promotion_active']:
        tags.append("Promo Active")
    if 'weather' in row:
        tags.append(f"Weather: {row['weather']}")
    return " | ".join(tags) if tags else "Normal"

def get_table_download_link(df):
    """Generates a link to download the dataframe as a CSV"""
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    filename = f"optimized_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href



def home_page():
    st.set_page_config(page_title="AI Dynamic Pricing", layout="wide")

    st.markdown("<h1 style='text-align: center;'>AI-Powered Dynamic Pricing Platform</h1>", unsafe_allow_html=True)

    # Center the image
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.image(
            r"D:\price_detection_2 (3)-new\price_detection_2\image.png",
            use_container_width=True,
            caption="Smart Pricing Solutions"
        )

    # Center the paragraph
    st.markdown("""
    <div style='text-align: center;'>
        <h2>Welcome to the Future of Pricing Optimization</h2>
        <p style='font-size: 18px; max-width: 800px; margin: auto;'>
            Our AI-powered platform helps businesses maximize profits through intelligent, data-driven pricing strategies.
            With real-time demand sensing, competitive insights, and shelf-life optimization, 
            you can stay ahead of market trends effortlessly.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
def about_page():
    # auth_header()
    st.title("About Price Optimization AI")
    
    st.subheader("Our Mission", divider='blue')
    st.markdown("""
    **To empower businesses with intelligent pricing solutions** that leverage artificial intelligence and domain expertise to drive sustainable growth, competitive advantage, and operational efficiency.
    """)
    
    st.subheader("Why Choose Our AI Pricing Solution?")
    st.markdown("""
    We transform pricing from a static business function into a dynamic growth engine. Our AI-powered platform continuously analyzes multiple data dimensions to deliver:
    - 📈 **Revenue growth** through optimized pricing strategies
    - � **Waste reduction** with intelligent inventory management
    - 🥇 **Competitive positioning** in dynamic markets
    - ⚙️ **Operational efficiency** through automation
    """)
    
    st.subheader("Core Features", divider='blue')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌦️ Weather-Adaptive Pricing
        **Dynamic adjustments based on forecast conditions:**  
        - Rainy → Increase demand for indoor products  
        - Sunny → Optimize seasonal/outdoor items  
        - Humidity/Snowy → Specialized category pricing  
        *Boost sales by aligning prices with consumption patterns*
        
        ### ⏱️ Expiry-Driven Discounts
        **Automatic margin-preserving markdowns:**  
        - Progressive discounting as expiry approaches  
        - Customizable discount curves by product category  
        - Waste reduction alerts and analytics  
        *Convert potential losses into revenue opportunities*
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Smart Promotion Engine
        **Intelligent stock-clearing system:**  
        - Automatic promotion triggering for slow-movers  
        - Demand forecasting for optimal discount depth  
        - Cross-selling opportunity identification  
        *Free up capital by optimizing inventory turnover*
        
        ### 🔍 Competitive Price Intelligence
        **Real-time market positioning:**  
        - Competitor price tracking and analysis  
        - Price gap identification and recommendations  
        - Dynamic repricing strategies by product segment  
        *Maintain price leadership while protecting margins*
        """)
    
    st.subheader("Our Value Proposition", divider='blue')
    st.markdown("""
    > "We combine cutting-edge machine learning algorithms with deep pricing expertise to create **data-driven strategies** that outperform traditional pricing methods. Our platform continuously learns from market signals to deliver **actionable insights** that drive measurable business outcomes."
    """)
    
    st.caption("Ready to transform your pricing strategy? [Contact us for a personalized demo]")

def optimizer_page():
    st.title("Price Optimization Dashboard")
    
    # Step 1: Upload CSV file
    uploaded_file = st.file_uploader("📤 Upload Product Data (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and show initial data
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Show initial data overview
            st.subheader("📊 Initial Data Overview")
            st.write(f"✅ Loaded {len(df)} products")
            
            # Show low stock items if column exists
            if 'stock_units' in df.columns:
                low_stock_df = df[df['stock_units'] < 30]
                st.subheader("⚠ Low Stock Items")
                st.write(f"Found {len(low_stock_df)} low stock items")
                st.dataframe(low_stock_df)
            
            st.markdown("---")
            
            # Step 2: Filter Products
            st.subheader("🔍 Filter Products")
            cols = st.columns(3)
            filter_params = {}
            
            with cols[0]:
                if 'category' in df.columns:
                    filter_params['category'] = st.multiselect(
                        "By Category", df['category'].unique())
                if 'demand_level' in df.columns:
                    filter_params['demand'] = st.multiselect(
                        "By Demand Level", df['demand_level'].unique())

            with cols[1]:
                if 'location' in df.columns:
                    filter_params['location'] = st.multiselect(
                        "By Location", df['location'].unique())
                if 'promotion_active' in df.columns:
                    filter_params['promotion'] = st.selectbox(
                        "Promotion Status", ['All', 'Active', 'Inactive'])
                
                # Weather filter
                if 'weather' in df.columns:
                    weather_options = ['Rainy', 'Humidity', 'Sunny', 'Cloudy', 'Snowy']
                    available_weather = [w for w in weather_options if w in df['weather'].unique()]
                    filter_params['weather'] = st.multiselect(
                        "By Weather Condition", 
                        available_weather,
                        help="1. Rainy\n2. Humidity\n3. Sunny\n4. Cloudy\n5. Snowy"
                    )

            with cols[2]:
                if 'stock_units' in df.columns:
                    min_stock = int(df['stock_units'].min())
                    max_stock = int(df['stock_units'].max())
                    filter_params['stock_range'] = st.slider(
                        "Stock Units Range",
                        min_stock,
                        max_stock,
                        (min_stock, max_stock)
                    )
                
                if 'days_to_expiry' in df.columns:
                    filter_params['expiry_range'] = st.slider(
                        "Days Until Expiry",
                        0, 365, (0, 365),
                        help="Products with <=30 days will get automatic discounts"
                    )

            if st.button("🚀 Apply Filters", type="primary"):
                filtered_df = df.copy()
                
                # Apply all filters
                for col in ['category', 'demand_level', 'location', 'weather']:
                    if col in filter_params and filter_params[col]:
                        filtered_df = filtered_df[filtered_df[col].isin(filter_params[col])]
                
                if 'promotion' in filter_params and filter_params['promotion'] != 'All':
                    active = filter_params['promotion'] == 'Active'
                    filtered_df = filtered_df[filtered_df['promotion_active'] == active]
                
                if 'stock_range' in filter_params:
                    min_stock, max_stock = filter_params['stock_range']
                    filtered_df = filtered_df[
                        (filtered_df['stock_units'] >= min_stock) & 
                        (filtered_df['stock_units'] <= max_stock)
                    ]
                
                if 'expiry_range' in filter_params:
                    min_days, max_days = filter_params['expiry_range']
                    filtered_df = filtered_df[
                        (filtered_df['days_to_expiry'] >= min_days) & 
                        (filtered_df['days_to_expiry'] <= max_days)
                    ]
                
                st.session_state.filtered_df = filtered_df
                st.session_state.filters_applied = True
                
                # Show filtered results
                st.subheader(f"📋 Filtered Results ({len(filtered_df)} products)")
                st.dataframe(filtered_df)
                
                # Step 3: Optimize Prices
                with st.spinner("🧠 Calculating optimal prices..."):
                    try:
                        predictions = st.session_state.predictor.predict_prices(filtered_df)
                        results_df = filtered_df.copy()
                        results_df['predicted_price'] = predictions
                        results_df['price_change_pct'] = (
                            (results_df['predicted_price'] - results_df['base_price']) / 
                            results_df['base_price'] * 100
                        ).round(1)
                        results_df['status'] = results_df.apply(display_status_tags, axis=1)
                        st.session_state.results_df = results_df
                        st.session_state.predicted = True
                        st.success("✅ Price optimization complete!")
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

            # Step 4: Show Optimized Results
            if st.session_state.predicted and st.session_state.results_df is not None:
                results_df = st.session_state.results_df
                st.subheader("📊 Optimization Results")
                
                # Summary metrics
                cols = st.columns(4)
                cols[0].metric("Products", len(results_df))
                cols[1].metric("Avg Price Change", f"{results_df['price_change_pct'].mean():.1f}%")
                cols[2].metric("Max Increase", f"{results_df['price_change_pct'].max():.1f}%")
                cols[3].metric("Max Decrease", f"{results_df['price_change_pct'].min():.1f}%")
                
                # Show optimized data
                st.dataframe(results_df[[
                    'product_id', 'base_price', 'predicted_price', 
                    'price_change_pct', 'status', 'weather'
                ]].sort_values('price_change_pct', ascending=False))
                
                # Download button
                st.markdown(get_table_download_link(results_df), unsafe_allow_html=True)
                
                # Step 5: Price Analysis
                st.subheader("📈 Price Analysis")
                tab1, tab2= st.tabs(["Price Changes", "Weather Impact"])
                
                with tab1:
                    fig = px.bar(
                        results_df.sort_values('price_change_pct'),
                        x='product_id',
                        y='price_change_pct',
                        title="Price Changes (%)",
                        color='price_change_pct',
                        color_continuous_scale=['red', 'green']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if 'weather' in results_df.columns:
                        weather_impact = results_df.groupby('weather')['price_change_pct'].mean().reset_index()
                        fig = px.bar(
                            weather_impact,
                            x='weather',
                            y='price_change_pct',
                            title="Average Price Change by Weather Condition",
                            color='weather',
                            color_discrete_map={
                                'Rainy': 'blue',
                                'Humidity': 'lightblue',
                                'Sunny': 'orange',
                                'Cloudy': 'gray',
                                'Snowy': 'white'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # with tab3:
                #     if 'days_to_expiry' in results_df.columns:
                #         # Bin days to expiry for better visualization
                #         results_df['expiry_bin'] = pd.cut(
                #             results_df['days_to_expiry'],
                #             bins=[0, 7, 15, 30, 60, 90, 365],
                #             labels=['0-7', '8-15', '16-30', '31-60', '61-90', '90+']
                #         )
                #         expiry_impact = results_df.groupby('expiry_bin')['price_change_pct'].mean().reset_index()
                #         fig = px.bar(
                #             expiry_impact,
                #             x='expiry_bin',
                #             y='price_change_pct',
                #             title="Average Price Change by Days to Expiry",
                #             color='price_change_pct',
                #             color_continuous_scale=['red', 'green']
                #         )
                #         st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("📤 Please upload a CSV file to get started")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "About", "Price Optimizer"]
    )
    
    if page == "Home":
        home_page()
    elif page == "About":
        about_page()
    elif page == "Price Optimizer":
        optimizer_page()

if __name__ == "__main__":
    main()