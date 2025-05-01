import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and prepare data
df = pd.read_csv("data_bodmas.csv")

categorical_cols = ['Donor_BloodType', 'Recipient_BloodType', 'Donor_Rh', 'Recipient_Rh', 'Organ_Type']
label_encoders = {}

# Encode categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train Random Forest model
X = df.drop(columns=['Donor_ID', 'Recipient_ID', 'Is_Match'])
y = df['Is_Match']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🫀 Organ Match Prediction")

with st.form("input_form"):
    donor_blood_type = st.selectbox("Donor Blood Type", ["A", "B", "AB", "O"])
    recipient_blood_type = st.selectbox("Recipient Blood Type", ["A", "B", "AB", "O"])
    donor_rh = st.selectbox("Donor Rh", ["+", "-"])
    recipient_rh = st.selectbox("Recipient Rh", ["+", "-"])
    organ_type = st.selectbox("Organ Type", ["Kidney", "Heart", "Liver", "Lung", "Pancreas"])

    donor_age = st.number_input("Donor Age", min_value=0, max_value=120, value=30)
    recipient_age = st.number_input("Recipient Age", min_value=0, max_value=120, value=40)

    donor_lat = st.number_input("Donor Latitude", value=25.5)
    donor_lon = st.number_input("Donor Longitude", value=80.1)
    recipient_lat = st.number_input("Recipient Latitude", value=25.6)
    recipient_lon = st.number_input("Recipient Longitude", value=80.0)

    submit = st.form_submit_button("Predict Match")

if submit:
    # Prepare input for prediction
    new_data = {
        "Donor_BloodType": donor_blood_type,
        "Recipient_BloodType": recipient_blood_type,
        "Donor_Rh": donor_rh,
        "Recipient_Rh": recipient_rh,
        "Organ_Type": organ_type,
        "Donor_Age": donor_age,
        "Recipient_Age": recipient_age,
        "Donor_Lat": donor_lat,
        "Donor_Lon": donor_lon,
        "Recipient_Lat": recipient_lat,
        "Recipient_Lon": recipient_lon
    }

    input_df = pd.DataFrame([new_data])

    # Encode input
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = le.transform([input_df[col][0]])

    prediction = model.predict(input_df)[0]
    result = "✅ Match" if prediction == 1 else "❌ No Match"

    st.success(f"Prediction Result: {result}")
