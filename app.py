import streamlit as st
import numpy as np
import joblib

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="FinSeg",
    page_icon="💳",
    layout="wide",
)

# --------------------- LOAD MODELS (CACHED) ---------------------
@st.cache_resource
def load_models():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_models()

# --------------------- CLEAN DARK CSS ---------------------
st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #ffffff;
    }
    .sub-title {
        font-size: 16px;
        color: #bbbbbb;
        margin-bottom: 5px;
    }
    .section {
        width: 75%;
    }
    .prediction-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #111111;
        border: 1px solid #444444;
        color: #ffffff;
        margin-top: 15px;
        font-size: 18px;
        width: 75%;
    }
</style>
""", unsafe_allow_html=True)

# --------------------- HEADER ---------------------
st.markdown("<h1 class='main-title'> 🗂️ Customer Financial Segmentation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter customer information to generate the predicted financial behavior category.</p>", unsafe_allow_html=True)

# --------------------- INPUT SECTION ---------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("👤 Customer Profile")

# Layout columns
col1, col2 = st.columns(2)

with col1:
    st.write("### 💼 Financial Information")
    income = st.number_input("Income", min_value=0)
    expenses = st.number_input("Expenses", min_value=0)
    savings_rate = st.number_input("Savings Rate (0.0 - 2.5)", min_value=0.0, max_value=2.5, format="%.3f")

with col2:
    st.write("### 💳 Credit Details")
    credit_cards = st.number_input("Credit Cards Count", min_value=0, max_value=10)
    credit_utilization = st.number_input("Credit Utilization (%)", min_value=0, max_value=100)
    emi_count = st.number_input("Ongoing EMIs", min_value=0, max_value=10)

st.write("### ✨ Lifestyle")
online_shopping_spend = st.number_input("Online Shopping Spend", min_value=0)
age = st.number_input("Age", min_value=18, max_value=100)

predict_btn = st.button("Generate Category", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------- PREDICTION LOGIC ---------------------
prediction_made = False

if predict_btn:

    # Basic input validation
    if expenses > income:
        st.warning("⚠️ Expenses exceed income — prediction may be inaccurate.")
    if savings_rate > 1.0:
        st.info("💡 Savings rate above 1.0 means saving more than total income. Ensure this is intended.")

    input_data = np.array([[income, expenses, savings_rate, credit_cards,
                            credit_utilization, emi_count, online_shopping_spend, age]])

    scaled_input = scaler.transform(input_data)
    pred = model.predict(scaled_input)[0]
    category = label_encoder.inverse_transform([pred])[0]

    st.markdown(
        f"<div class='prediction-box'>🎯 <b>Predicted Category:</b> {category}</div>",
        unsafe_allow_html=True,
    )
    prediction_made = True

else:
    st.info("Fill in the details and click **Generate Category**.")

# --------------------- CATEGORY GUIDE ---------------------
if prediction_made:
    st.markdown("---")
    st.subheader("📘 Category Guide")
    st.write("""
### › **Saver**  
Individuals who maintain low spending, are budget conscious, and manage finances with caution.

### › **Balanced**  
Moderate and stable spending patterns with a healthy balance between expenses and savings.

### › **Spender**  
High spending tendencies, often linked to premium lifestyle and higher credit utilization.
""")
