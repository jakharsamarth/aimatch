import pandas as pd
import xgboost as xgb
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load dataset and train model
df = pd.read_csv("data_bodmas.csv")

categorical_cols = ['Donor_BloodType', 'Recipient_BloodType', 'Donor_Rh', 'Recipient_Rh', 'Organ_Type']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['Donor_ID', 'Recipient_ID', 'Is_Match'])
y = df['Is_Match']
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("Organ Match Prediction")

donor_blood = st.selectbox("Donor Blood Type", label_encoders['Donor_BloodType'].classes_)
recipient_blood = st.selectbox("Recipient Blood Type", label_encoders['Recipient_BloodType'].classes_)
donor_rh = st.selectbox("Donor Rh", label_encoders['Donor_Rh'].classes_)
recipient_rh = st.selectbox("Recipient Rh", label_encoders['Recipient_Rh'].classes_)
organ_type = st.selectbox("Organ Type", label_encoders['Organ_Type'].classes_)

donor_age = st.number_input("Donor Age", min_value=0, max_value=120)
recipient_age = st.number_input("Recipient Age", min_value=0, max_value=120)
donor_lat = st.number_input("Donor Latitude")
donor_lon = st.number_input("Donor Longitude")
recipient_lat = st.number_input("Recipient Latitude")
recipient_lon = st.number_input("Recipient Longitude")

if st.button("Predict Match"):
    input_data = {
        "Donor_BloodType": donor_blood,
        "Recipient_BloodType": recipient_blood,
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

    # Encode input
    for col in categorical_cols:
        le = label_encoders[col]
        input_data[col] = le.transform([input_data[col]])[0]

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    result = "✅ Match" if prediction == 1 else "❌ No Match"
    st.subheader(f"Prediction Result: {result}")
