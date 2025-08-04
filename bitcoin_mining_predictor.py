import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Page configuration
st.set_page_config(
    page_title="Bitcoin Mining Location Predictor",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #f7931a;
    text-align: center;
    margin-bottom: 2rem;
}
.company-intro {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 2rem 0;
    text-align: center;
}
.prediction-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.metric-good {
    color: #28a745;
    font-weight: bold;
}
.metric-warning {
    color: #ffc107;
    font-weight: bold;
}
.metric-danger {
    color: #dc3545;
    font-weight: bold;
}
.logo-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

def display_company_header():
    """Display Exponential Science logo and company introduction"""
    
    # Create columns for logo positioning - logo in right corner
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col3:  # Changed from col2 to col3 (right column)
        # Add your logo here
        try:
            st.image("exponential_science_logo.png", width=100)  # Made smaller for corner
        except FileNotFoundError:
            # Fallback placeholder if logo file not found
            st.markdown("""
            <div style="text-align: right;">
                <div style="background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%); 
                           width: 200px; height: 60px; border-radius: 10px; 
                           display: flex; align-items: center; justify-content: center;
                           color: white; font-size: 16px; font-weight: bold;
                           margin-left: auto;">
                    📊 EXPONENTIAL SCIENCE
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown("<div style='text-align: right;'><strong>🏢 EXPONENTIAL SCIENCE</strong></div>", unsafe_allow_html=True)
    
    # Company introduction
    st.markdown("""
    <div class="company-intro">
        <h2>🚀 About Exponential Science</h2>
        <p style="font-size: 1.2rem; margin: 1rem 0;">
            <strong>Exponential Science</strong> is a cutting-edge data science and analytics company specializing in 
            predictive modeling and advanced analytics for emerging technologies and financial markets.
        </p>
        <div style="display: flex; justify-content: space-around; margin: 2rem 0; flex-wrap: wrap;">
            <div style="margin: 0.5rem;">
                <strong>🎯 Expertise</strong><br>
                Cryptocurrency Analytics<br>
            </div>
            <div style="margin: 0.5rem;">
                <strong>🌍 Focus Areas</strong><br>
                Blockchain Technology<br>
                Market Analysis<br>
            </div>
            <div style="margin: 0.5rem;">
                <strong>💡 Innovation</strong><br>
                Real-time Analytics<br>
                Business Intelligence
            </div>
        </div>
        <p style="font-style: italic; margin-top: 1.5rem;">
            "Transforming complex data into actionable insights for the digital economy"
        </p>
    </div>
    """, unsafe_allow_html=True)

def load_models():
    """Load your trained models (you'll need to save them first)"""
    # For now, we'll create mock models - replace with your actual saved models
    
    # Mock Linear Regression coefficients (use your actual coefficients)
    lr_coef = np.array([-9.315, -3.784, 2.700, 1.684, 0.691, -0.276])  # Your actual coefficients
    lr_intercept = 10.0  # Adjust based on your model
    
    # Mock Random Forest (replace with your actual model)
    rf_feature_importance = {
        'policy_mining_potential': 0.193,
        'regulatory_score': 0.184,
        'expected_ban_impact': 0.169,
        'renewable_pct': 0.160,
        'electricity_price_kwh': 0.151
    }
    
    return lr_coef, lr_intercept, rf_feature_importance

def predict_hashrate(features, lr_coef, lr_intercept):
    """Predict hashrate using linear regression coefficients"""
    feature_array = np.array([
        features['renewable_pct'],
        features['electricity_price_kwh'] * 100,  # Scale for coefficients
        features['regulatory_score'],
        features['expected_ban_impact'],
        features['months_since_ban'],
        features['post_china_ban']
    ])
    
    prediction = np.dot(feature_array, lr_coef) + lr_intercept
    return max(0, min(prediction, 100))  # Bound between 0-100%

def predict_major_location(features, rf_importance):
    """Predict major location probability using feature importance"""
    # Simplified prediction based on weighted features
    score = (
        (features['renewable_pct'] / 100) * rf_importance['renewable_pct'] +
        (features['regulatory_score'] / 10) * rf_importance['regulatory_score'] +
        (1 - features['electricity_price_kwh'] / 0.3) * rf_importance['electricity_price_kwh'] +
        ((features['expected_ban_impact'] + 50) / 100) * rf_importance['expected_ban_impact']
    )
    
    # Convert to probability
    probability = min(max(score, 0), 1)
    return probability

def main():
    # Display company header and logo FIRST
    display_company_header()
    
    # Header
    st.markdown('<h1 class="main-header">⛏️ Bitcoin Mining Location Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎓 Master's Dissertation: Predictive Model for Bitcoin Mining Location Analysis
    **Predict mining potential and investment attractiveness for any location worldwide**
    
    *This tool uses machine learning models trained on global Bitcoin mining data to predict 
    hashrate potential and classify locations as major mining hubs.*
    """)
    
    # Load models
    lr_coef, lr_intercept, rf_importance = load_models()
    
    # Sidebar for inputs
    st.sidebar.header("🌍 Location Characteristics")
    st.sidebar.markdown("*Enter the characteristics of your potential mining location:*")
    
    # Input fields
    location_name = st.sidebar.text_input("📍 Location Name", value="My Mining Location")
    
    st.sidebar.subheader("💡 Energy & Cost Factors")
    renewable_pct = st.sidebar.slider(
        "Renewable Energy %", 
        min_value=0, max_value=100, value=50, step=5,
        help="Percentage of electricity from renewable sources"
    )
    
    electricity_price = st.sidebar.slider(
        "Electricity Price ($/kWh)", 
        min_value=0.02, max_value=0.30, value=0.08, step=0.01,
        help="Industrial electricity rate in USD per kWh"
    )
    
    st.sidebar.subheader("🏛️ Regulatory Environment")
    regulatory_score = st.sidebar.slider(
        "Regulatory Score (1-10)", 
        min_value=1, max_value=10, value=7, step=1,
        help="1=Very Restrictive (China post-ban), 10=Very Favorable"
    )
    
    st.sidebar.subheader("📊 Policy Impact")
    expected_ban_impact = st.sidebar.slider(
        "Expected Policy Impact (pp)", 
        min_value=-50, max_value=20, value=0, step=5,
        help="Expected change in hashrate due to policy changes (+ = beneficial, - = harmful)"
    )
    
    months_since_ban = st.sidebar.slider(
        "Months Since China Ban", 
        min_value=0, max_value=24, value=18, step=1,
        help="Time since China's mining ban (for adaptation analysis)"
    )
    
    post_china_ban = st.sidebar.selectbox(
        "Analysis Period", 
        options=[1, 0], 
        format_func=lambda x: "Post China Ban" if x == 1 else "Pre China Ban",
        index=0
    )
    
    # Prepare features
    features = {
        'renewable_pct': renewable_pct,
        'electricity_price_kwh': electricity_price,
        'regulatory_score': regulatory_score,
        'expected_ban_impact': expected_ban_impact,
        'months_since_ban': months_since_ban,
        'post_china_ban': post_china_ban
    }
    
    # Make predictions
    if st.sidebar.button("🔮 Generate Predictions", type="primary"):
        
        # Predictions
        predicted_hashrate = predict_hashrate(features, lr_coef, lr_intercept)
        major_location_prob = predict_major_location(features, rf_importance)
        
        # Investment recommendation logic
        if predicted_hashrate > 15 and major_location_prob > 0.7:
            investment_rec = "🟢 STRONG BUY"
            rec_color = "metric-good"
        elif predicted_hashrate > 8 and major_location_prob > 0.5:
            investment_rec = "🟡 BUY"
            rec_color = "metric-warning"
        elif predicted_hashrate > 3:
            investment_rec = "🟠 HOLD"
            rec_color = "metric-warning"
        else:
            investment_rec = "🔴 AVOID"
            rec_color = "metric-danger"
        
        # Main content area - Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🎯 Predicted Hashrate Share",
                value=f"{predicted_hashrate:.1f}%",
                delta=f"vs Global Average (10.0%): {predicted_hashrate-10:.1f}pp"
            )
        
        with col2:
            st.metric(
                label="🏢 Major Location Probability", 
                value=f"{major_location_prob*100:.1f}%",
                delta="High" if major_location_prob > 0.6 else "Moderate" if major_location_prob > 0.4 else "Low"
            )
        
        with col3:
            st.metric(
                label="💰 Investment Recommendation",
                value=investment_rec
            )
        
        # Detailed Analysis
        st.markdown("---")
        st.subheader("📊 Detailed Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction Breakdown", "📈 Benchmarking", "🔍 Sensitivity Analysis", "🌍 Comparison"])
        
        with tab1:
            st.markdown("### Prediction Breakdown")
            
            # Factor contribution analysis
            factor_contributions = {
                'Renewable Energy': renewable_pct * 0.1,
                'Electricity Cost': (0.3 - electricity_price) * 30,
                'Regulatory Environment': regulatory_score * 1.5,
                'Policy Impact': expected_ban_impact * 0.2,
                'Market Timing': months_since_ban * 0.3
            }
            
            contrib_df = pd.DataFrame(list(factor_contributions.items()), columns=['Factor', 'Contribution'])
            
            fig_contrib = px.bar(
                contrib_df, 
                x='Contribution', 
                y='Factor', 
                orientation='h',
                title="Factor Contributions to Hashrate Prediction",
                color='Contribution',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
        
        with tab2:
            st.markdown("### Benchmarking Against Known Locations")
            
            # Benchmark data (from your EDA)
            benchmark_data = {
                'Location': ['China (Pre-ban)', 'United States', 'Kazakhstan', 'Your Location'],
                'Hashrate_Share': [61.5, 13.8, 7.5, predicted_hashrate],
                'Renewable_Energy': [28.5, 57.5, 12.1, renewable_pct],
                'Regulatory_Score': [3.0, 7.4, 6.5, regulatory_score],
                'Type': ['Historical', 'Current', 'Current', 'Prediction']
            }
            
            benchmark_df = pd.DataFrame(benchmark_data)
            
            fig_bench = px.scatter(
                benchmark_df, 
                x='Renewable_Energy', 
                y='Hashrate_Share',
                size='Regulatory_Score',
                color='Type',
                text='Location',
                title="Your Location vs. Global Mining Hubs"
            )
            fig_bench.update_traces(textposition="top center")
            st.plotly_chart(fig_bench, use_container_width=True)
        
        with tab3:
            st.markdown("### Sensitivity Analysis")
            st.markdown("*How sensitive are predictions to changes in key factors?*")
            
            # Sensitivity analysis
            base_hashrate = predicted_hashrate
            
            sensitivity_data = []
            factors = ['renewable_pct', 'electricity_price_kwh', 'regulatory_score']
            
            for factor in factors:
                for change in [-20, -10, 0, 10, 20]:
                    modified_features = features.copy()
                    
                    if factor == 'renewable_pct':
                        modified_features[factor] = max(0, min(100, renewable_pct + change))
                    elif factor == 'electricity_price_kwh':
                        modified_features[factor] = max(0.02, min(0.30, electricity_price + change/100))
                    elif factor == 'regulatory_score':
                        modified_features[factor] = max(1, min(10, regulatory_score + change/10))
                    
                    new_prediction = predict_hashrate(modified_features, lr_coef, lr_intercept)
                    
                    sensitivity_data.append({
                        'Factor': factor,
                        'Change': change,
                        'New_Prediction': new_prediction,
                        'Impact': new_prediction - base_hashrate
                    })
            
            sens_df = pd.DataFrame(sensitivity_data)
            
            fig_sens = px.line(
                sens_df, 
                x='Change', 
                y='Impact', 
                color='Factor',
                title="Sensitivity of Hashrate Prediction to Factor Changes",
                labels={'Impact': 'Change in Predicted Hashrate (pp)'}
            )
            st.plotly_chart(fig_sens, use_container_width=True)
        
        with tab4:
            st.markdown("### Multi-Location Comparison")
            
            # Allow users to compare multiple scenarios
            st.markdown("*Compare your location against different scenarios:*")
            
            scenarios = {
                'Conservative': {'renewable_pct': 30, 'electricity_price_kwh': 0.12, 'regulatory_score': 6},
                'Aggressive': {'renewable_pct': 80, 'electricity_price_kwh': 0.05, 'regulatory_score': 9},
                'Current': features
            }
            
            comparison_data = []
            for scenario_name, scenario_features in scenarios.items():
                scenario_features.update({
                    'expected_ban_impact': expected_ban_impact,
                    'months_since_ban': months_since_ban,
                    'post_china_ban': post_china_ban
                })
                
                hashrate_pred = predict_hashrate(scenario_features, lr_coef, lr_intercept)
                major_prob = predict_major_location(scenario_features, rf_importance)
                
                comparison_data.append({
                    'Scenario': scenario_name,
                    'Predicted_Hashrate': hashrate_pred,
                    'Major_Location_Prob': major_prob * 100,
                    'Renewable_Energy': scenario_features['renewable_pct'],
                    'Electricity_Price': scenario_features['electricity_price_kwh'],
                    'Regulatory_Score': scenario_features['regulatory_score']
                })
            
            comp_df = pd.DataFrame(comparison_data)
            
            fig_comp = px.bar(
                comp_df, 
                x='Scenario', 
                y='Predicted_Hashrate',
                title="Scenario Comparison: Predicted Hashrate",
                color='Predicted_Hashrate',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.dataframe(comp_df, use_container_width=True)
        
        # Business Insights
        st.markdown("---")
        st.subheader("💼 Business Insights & Recommendations")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("#### 🎯 Key Strengths")
            strengths = []
            if renewable_pct > 70:
                strengths.append("• High renewable energy (sustainability advantage)")
            if electricity_price < 0.08:
                strengths.append("• Low electricity costs (cost advantage)")
            if regulatory_score > 7:
                strengths.append("• Favorable regulatory environment")
            if expected_ban_impact > 0:
                strengths.append("• Benefits from policy changes")
            
            if strengths:
                for strength in strengths:
                    st.markdown(strength)
            else:
                st.markdown("• Consider improving key factors for better performance")
        
        with insights_col2:
            st.markdown("#### ⚠️ Risk Factors")
            risks = []
            if renewable_pct < 30:
                risks.append("• Low renewable energy (sustainability risk)")
            if electricity_price > 0.15:
                risks.append("• High electricity costs (profitability risk)")
            if regulatory_score < 5:
                risks.append("• Uncertain regulatory environment")
            if expected_ban_impact < -10:
                risks.append("• Negative policy impact expected")
            
            if risks:
                for risk in risks:
                    st.markdown(risk)
            else:
                st.markdown("• Low risk profile identified")
        
        # Action items
        st.markdown("#### 🚀 Recommended Actions")
        if investment_rec == "🟢 STRONG BUY":
            st.success("Excellent location for Bitcoin mining investment. Consider immediate deployment.")
        elif investment_rec == "🟡 BUY":
            st.info("Good mining potential. Consider investment with standard due diligence.")
        elif investment_rec == "🟠 HOLD":
            st.warning("Moderate potential. Monitor regulatory changes and energy developments.")
        else:
            st.error("High risk location. Consider alternative locations or wait for improvements.")

    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📖 About This Tool
    This predictive model was developed as part of a Master's dissertation analyzing Bitcoin mining location decisions. 
    The models are trained on historical data including the impact of China's mining ban and various location characteristics.
    
    **Models Used:**
    - Linear Regression for hashrate prediction (R² = 0.647)
    - Random Forest for major location classification (76.7% accuracy)
    - Policy impact analysis based on empirical findings
    
    **Disclaimer:** This tool is for educational and research purposes. Investment decisions should consider additional factors and professional advice.
    """)

if __name__ == "__main__":
    main()