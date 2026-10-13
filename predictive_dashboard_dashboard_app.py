import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import google.generativeai as genai

st.set_page_config(page_title="GenAI Data Analyst Portfolio", layout="wide")

# Title Section
st.title("📈 Enterprise Predictive Sales Dashboard (AI-Enhanced)")
st.markdown("""
*Built by a GenAI Data Analyst to bridge machine learning pipelines with Large Language Models (LLMs) for automated business decision-making.*
***
""")

# ====================================================================
# 1. Dynamic Data Uploader Section (New Feature)
# ====================================================================
st.subheader("📁 Upload Enterprise Dataset")
uploaded_file = st.file_uploader("Choose a CSV file to train the ML Model", type=["csv"])

@st.cache_data
def load_custom_data(file_source):
    df = pd.read_csv(file_source)
  
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df


if uploaded_file is not None:
    df = load_custom_data(uploaded_file)
    st.success("✅ Custom CSV uploaded and analyzed successfully!")
else:
    df = load_custom_data("company_sales.csv")
    st.info("ℹ️ Using default 'company_sales.csv' database. Upload your own above.")


# 2. Sidebar Controls
st.sidebar.header("🔧 Live Simulation Controls")
input_marketing = st.sidebar.slider("Marketing Budget ($)", 500, 5000, 2500, step=50)
input_discount = st.sidebar.slider("Active Discount (%)", 0, 25, 10, step=1)
input_comp_price = st.sidebar.slider("Competitor Price ($)", 15, 30, 22, step=1)

# 3. Machine Learning Pipeline
X = df[["Marketing_Spend", "Discount_Percentage", "Competitor_Price"]]
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

accuracy = r2_score(y_test, predictions := model.predict(X_test))
live_pred = model.predict([[input_marketing, input_discount, input_comp_price]])[0]

# 4. KPI Metrics Cards
col1, col2, col3 = st.columns(3)
col1.metric(label="Total Historical Revenue", value=f"${df['Sales'].sum():,.2f}")
col2.metric(label="Simulated Sales Forecast", value=f"${live_pred:,.2f}", delta=f"{((live_pred - df['Sales'].mean())/df['Sales'].mean())*100:.1f}% vs Baseline Avg")
col3.metric(label="Model Confidence (R² Score)", value=f"{accuracy:.2f}")

st.markdown("***")

# 5. Visualizations & Data Ledger
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("📊 Marketing Investment vs. Revenue Yield")
    chart_data = df.sort_values(by="Marketing_Spend")
    st.line_chart(data=chart_data, x="Marketing_Spend", y="Sales", height=300)

with col_right:
    st.subheader("📋 Operational Data Ledger")
    formatted_df = df.tail(10).copy()
    for col in ["Marketing_Spend", "Competitor_Price", "Sales"]:
        formatted_df[col] = formatted_df[col].map("${:,.2f}".format)
    formatted_df["Discount_Percentage"] = formatted_df["Discount_Percentage"].map("{:.1f}%".format)
    st.dataframe(formatted_df, use_container_width=True)

st.markdown("***")

# 🤖 6. THE KILLER FEATURE: GenAI Executive Insights Consultant
st.subheader("🤖 GenAI Executive Strategy Consultant")
st.markdown("Click the button below to allow the integrated LLM to analyze the simulated metrics and generate automated business recommendations.")

# API Key Input box for security and portability
api_key_input = st.text_input("Enter your Gemini API Key to unlock AI Consultant:", type="password")

if st.button("🚀 Generate AI Executive Report"):
    if not api_key_input:
        st.error("Please enter a valid Gemini API Key first!")
    else:
        with st.spinner("AI is analyzing dashboard parameters and historical data..."):
            try:
                # Configure Gemini API
                genai.configure(api_key=api_key_input)
                ai_model = genai.GenerativeModel('gemini-3.8-flash')

                
                # Context prompt mapping data to AI
                prompt = f"""
                You are an expert McKinsey Business Intelligence Consultant and Senior Data Analyst.
                Analyze the following simulated dashboard metrics and provide a professional, executive summary:
                
                - Current Simulated Marketing Budget: ${input_marketing}
                - Active Customer Discount Percentage: {input_discount}%
                - Competitor's Product Price: ${input_comp_price}
                - Machine Learning Predicted Sales Output: ${live_pred:.2f}
                - Historical Baseline Sales Average: ${df['Sales'].mean():.2f}
                
                Provide 3 clear, high-impact bullet points outlining:
                1. A brief analysis of how the current competitor pricing impacts our predicted sales.
                2. A tactical recommendation on whether we should increase/decrease marketing or discounts based on these simulated inputs.
                3. The potential risk if the competitor drops their price further.
                Keep it highly professional, short, and executive-ready.
                """
                
                response = ai_model.generate_content(prompt)
                st.success("Analysis Complete!")
                st.info(response.text)
                
            except Exception as e:
                st.error(f"Error connecting to Gemini API: {str(e)}")


# Sidebar Bottom Portfolio Section 

st.sidebar.markdown("***")  
st.sidebar.markdown("""
<div style='text-align: center; color: #888888; font-size: 13px; padding-top: 20px;'>
    <p style='margin-bottom: 2px;'><b>PROJECT TITLE:</b></p>
    <p style='color: #00ffcc; font-size: 14px; margin-top: 0px;'><b>GenAI Sales Analytics Engine</b></p>
    <p style='margin-bottom: 2px; margin-top: 15px;'><b>DEVELOPER:</b></p>
    <p style='color: #ffffff; font-size: 14px; margin-top: 0px;'>Md. Niyaz Alam | Data Analyst With GenAI</p>
</div>
""", unsafe_allow_html=True)


