import streamlit as st
import pandas as pd
import joblib
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading ---
@st.cache_resource
def load_model(path):
    """Loads the pre-trained model from the specified path."""
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model("knn_model.joblib")

# --- Main App ---
if model is None:
    st.stop()

# --- Header ---
st.title("💳 Credit Card Default Prediction")
st.markdown("This dashboard predicts whether a client is likely to default on their credit card payment based on their financial history.")

# --- Tabs ---
tab1, tab2 = st.tabs(["Prediction Dashboard", "About the App"])

with tab1:
    # --- Input Form using Columns and Containers ---
    st.header("👤 Client Information")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            limit_bal = st.number_input(
                "Credit Limit (NT$)", 
                min_value=10000, max_value=1000000, value=50000, step=1000,
                help="The amount of credit given in New Taiwan dollars."
            )
            sex = st.radio("Gender", ["Male", "Female"], horizontal=True)
        with col2:
            age = st.slider("Age", 21, 79, 30, help="Client's age in years.")
            education = st.selectbox(
                "Education Level", 
                ["Graduate School", "University", "High School", "Others"]
            )
        with col3:
            marriage = st.radio("Marital Status", ["Married", "Single", "Others"], horizontal=True)

    st.header("💳 Financial History (Last 6 Months)")
    
    # Using expanders to keep the UI clean
    with st.expander("Repayment Status", expanded=True):
        cols = st.columns(6)
        pay_labels = ["June 2025", "May 2025", "April 2025", "March 2025", "February 2025", "January 2025"]
        pay_keys = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        pay_values = [cols[i].slider(pay_labels[i], -2, 8, 0, key=f'pay_{i}') for i in range(6)]
    
    with st.expander("Bill Statement Amounts (NT$)"):
        cols = st.columns(6)
        bill_labels = [f"BILL_AMT{i}" for i in range(1, 7)]
        bill_values = [cols[i].number_input(bill_labels[i], 0, 1000000, 5000, key=f'bill_{i}') for i in range(6)]

    with st.expander("Previous Payment Amounts (NT$)"):
        cols = st.columns(6)
        pay_amt_labels = [f"PAY_AMT{i}" for i in range(1, 7)]
        pay_amt_values = [cols[i].number_input(pay_amt_labels[i], 0, 500000, 500, key=f'pay_amt_{i}') for i in range(6)]

    # --- Prediction Button and Logic ---
    st.divider()
    predict_button = st.button("🚀 Predict Default Risk", type="primary", use_container_width=True)

    if predict_button:
        # Mappings
        education_mapping = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
        marriage_mapping = {"Married": 1, "Single": 2, "Others": 3}
        sex_mapping = {"Male": 1, "Female": 2}

        # Create DataFrame
        user_input_data = pd.DataFrame({
            "LIMIT_BAL": [limit_bal], "SEX": [sex_mapping[sex]], "EDUCATION": [education_mapping[education]],
            "MARRIAGE": [marriage_mapping[marriage]], "AGE": [age],
            "PAY_0": [pay_values[0]], "PAY_2": [pay_values[1]], "PAY_3": [pay_values[2]],
            "PAY_4": [pay_values[3]], "PAY_5": [pay_values[4]], "PAY_6": [pay_values[5]],
            "BILL_AMT1": [bill_values[0]], "BILL_AMT2": [bill_values[1]], "BILL_AMT3": [bill_values[2]],
            "BILL_AMT4": [bill_values[3]], "BILL_AMT5": [bill_values[4]], "BILL_AMT6": [bill_values[5]],
            "PAY_AMT1": [pay_amt_values[0]], "PAY_AMT2": [pay_amt_values[1]], "PAY_AMT3": [pay_amt_values[2]],
            "PAY_AMT4": [pay_amt_values[3]], "PAY_AMT5": [pay_amt_values[4]], "PAY_AMT6": [pay_amt_values[5]],
        })

        with st.spinner('🤖 Analyzing client data...'):
            time.sleep(1) # Simulate processing time
            
            # Predict probability
            prediction_proba = model.predict_proba(user_input_data)[0]
            default_probability = prediction_proba[1] # Probability of class '1' (default)

        # --- Display Results ---
        st.subheader("Prediction Result")
        result_col1, result_col2 = st.columns([2, 3])

        with result_col1:
            st.metric("Default Probability", f"{default_probability:.2%}")

            if default_probability > 0.5:
                st.error("High Risk of Default", icon="🚨")
            elif default_probability > 0.2:
                st.warning("Moderate Risk of Default", icon="⚠️")
            else:
                st.success("Low Risk of Default", icon="✅")

        with result_col2:
            st.write("Probability Distribution:")
            st.progress(default_probability)
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <span style="color: green;">{"%.1f" % ((1-default_probability)*100)}% Chance of No Default</span> | 
                    <span style="color: red;">{"%.1f" % (default_probability*100)}% Chance of Default</span>
                </div>
                """, unsafe_allow_html=True
            )

with tab2:
    st.header("About the Application")
    st.markdown("""
    This interactive dashboard is designed to predict the probability of a credit card holder defaulting on their next payment. 
    It leverages a **K-Nearest Neighbors (KNN)** machine learning model trained on a publicly available dataset.

    #### **How to Use:**
    1.  **Enter Client Information:** Fill in the details in the 'Prediction Dashboard' tab.
    2.  **Financial History:** Provide the repayment status, bill amounts, and payment amounts for the last six months.
    3.  **Predict:** Click the 'Predict Default Risk' button to see the result.

    #### **Dataset:**
    The model was trained on the "Default of Credit Card Clients" dataset from the UCI Machine Learning Repository. 
    It contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

    #### **Disclaimer:**
    This prediction is for informational and educational purposes only. It should not be used as the sole basis for making financial decisions.
    """)
    # st.image("https://i.imgur.com/v233F5A.png", caption="Machine Learning Model Workflow", use_column_width=True)