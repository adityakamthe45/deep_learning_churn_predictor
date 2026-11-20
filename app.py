import pickle
 
import pandas as pd
import streamlit as st
import tensorflow as tf
 
st.set_page_config(page_title="ChurnSense by Aditya", page_icon="📊", layout="wide")
 

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.h5")
    with open("onehot_encoder_geo.pkl", "rb") as f:
        geo_encoder = pickle.load(f)
    with open("label_encoder_gender.pkl", "rb") as f:
        gender_encoder = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, geo_encoder, gender_encoder, scaler
 
 
model, geo_encoder, gender_encoder, scaler = load_artifacts()
 

st.title("🏦 Customer Churn Prediction Dashboard")
st.write("Predict whether a customer is likely to leave the bank using a trained Deep Learning model.")
st.divider()
 

st.sidebar.header("Customer Information")
geography = st.sidebar.selectbox("🌍 Geography", geo_encoder.categories_[0])
gender = st.sidebar.selectbox("👤 Gender", gender_encoder.classes_)
credit_score = st.sidebar.number_input("💳 Credit Score", 0, 900, 650)
age = st.sidebar.slider("🎂 Age", 18, 92, 35)
tenure = st.sidebar.slider("📅 Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("💰 Balance", value=50000.0)
num_of_products = st.sidebar.slider("📦 Number of Products", 1, 4, 2)
has_cr_card = st.sidebar.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.sidebar.selectbox("✅ Active Member", [0, 1])
estimated_salary = st.sidebar.number_input("💵 Estimated Salary", value=100000.0)
 

c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", age)
c2.metric("Credit Score", credit_score)
c3.metric("Balance", f"${balance:,.0f}")
c4.metric("Salary", f"${estimated_salary:,.0f}")
st.divider()
 
if st.button("🔮 Predict Churn Probability", use_container_width=True):
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoder.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    })
 
    geo_df = pd.DataFrame(
        geo_encoder.transform([[geography]]).toarray(),
        columns=geo_encoder.get_feature_names_out(["Geography"]),
    )
    input_data = pd.concat([input_data, geo_df], axis=1)
 
    prob = float(model.predict(scaler.transform(input_data))[0][0])
 
    st.subheader("Prediction Results")
    left, right = st.columns([2, 1])
 
    with left:
        st.progress(prob)
        st.metric("Churn Probability", f"{prob:.2%}")
 
    with right:
        if prob >= 0.5:
            st.error("⚠️ **HIGH RISK**\n\nCustomer Likely to Churn")
        else:
            st.success("✅ **LOW RISK**\n\nCustomer Likely to Stay")
 
    st.divider()
    st.subheader("Model Confidence")
    st.info(f"Probability of Churn: **{prob:.2%}**\n\nProbability of Retention: **{1 - prob:.2%}**")