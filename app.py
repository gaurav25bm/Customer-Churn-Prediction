import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page layout and title
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# ==========================================
# LOAD ARTIFACTSa
# ==========================================

@st.cache_resource
def load_assets():
    with open("final_rf_churn_model.pkl", "rb") as model_file:
        loaded_stuff = pickle.load(model_file)
        
    # --- FIXED: Extract the actual model if it's trapped inside a container ---
    if isinstance(loaded_stuff, list):
        # If it's a list, the model is usually the first item
        model = loaded_stuff[0] 
    elif isinstance(loaded_stuff, dict):
        # If it's a dictionary from our model selection step, let's grab the actual estimator
        # Try common keys, or fallback to the first value
        model = loaded_stuff.get('RandomForest', list(loaded_stuff.values())[0])
    else:
        # It's already a single model object
        model = loaded_stuff

    with open("encoders.pkl", "rb") as encoder_file:
        encoders = pickle.load(encoder_file)
        
    return model, encoders
try:
    model, encoders = load_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Make sure pkl files are in the same folder.")

# ==========================================
# APP HEADER
# ==========================================
st.title("📊 Customer Churn Prediction App")
st.markdown("""
Predict whether a telecom customer will leave or stay based on demographics, 
account configurations, and usage metrics.
""")
st.write("---")

# ==========================================
# USER INPUT INTERFACE
# ==========================================
st.subheader("👤 Customer Demographic & Account Profile")

# Create a clean layout with 2 columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=12)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing?", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0, step=1.0)
    total_charges = st.slider("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=50.0)
    
    

st.subheader("🌐 Services Subscribed")
col3, col4 = st.columns(2)

with col3:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

with col4:
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# ==========================================
# ==========================================
# PROCESS INPUTS & PREDICT
# ==========================================
if st.button("🔮 Predict Churn Status", type="primary", use_container_width=True):
    
    # ⚙️ MANUAL ENCODING MAPS (To match what your Random Forest model expects)
    binary_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}
    
    # Map multi-class categories manually if LabelEncoder/OneHot was used
    # Assuming standard Telco mapping order:
    multiple_lines_map = {"No phone service": 0, "No": 1, "Yes": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    service_map = {"No": 0, "Yes": 1, "No internet service": 2}
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    payment_map = {
        "Electronic check": 0, 
        "Mailed check": 1, 
        "Bank transfer (automatic)": 2, 
        "Credit card (automatic)": 3
    }

    try:
        # Convert all UI selections into numeric values
        features = [
            gender_map[gender],
            1 if senior_citizen == "Yes" else 0,
            binary_map[partner],
            binary_map[dependents],
            int(tenure),
            binary_map[phone_service],
            multiple_lines_map[multiple_lines],
            internet_map[internet_service],
            service_map[online_security],
            service_map[online_backup],
            service_map[device_protection],
            service_map[tech_support],
            service_map[streaming_tv],
            service_map[streaming_movies],
            contract_map[contract],
            binary_map[paperless],
            payment_map[payment_method],
            float(monthly_charges),
            float(total_charges)
        ]
        
        # Reshape data for a single prediction array
        input_array = np.array(features).reshape(1, -1)
        
        # Generate prediction and probabilities
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1] # Probability of Churn (Class 1)
        
        st.write("---")
        if prediction == 1:
            st.error(f"### 🚨 High Risk Customer: Likely to Churn ({probability*100:.1f}% confidence)")
            st.warning("💡 **Retention Strategy Recommendation:** Offer a long-term contract discount or tech support check-in.")
        else:
            st.success(f"### 🎉 Low Risk Customer: Likely to Stay ({(1 - probability)*100:.1f}% confidence)")
            
    except Exception as err:
        st.error(f"Prediction failed: {err}")
        st.info("If it still fails, your notebook might have used a different column order or encoding numbers. Check your training feature list!")