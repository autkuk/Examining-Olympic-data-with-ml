
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_user_input():
    st.sidebar.header("Kullanıcı Bilgileri")
    Age = st.sidebar.number_input("Yaş", min_value=10, max_value=100)
    Height = st.sidebar.number_input("Boy (cm)", min_value=140, max_value=220)
    Weight = st.sidebar.number_input("Kilo (kg)", min_value=40, max_value=150)

    Sex = st.sidebar.selectbox("Cinsiyet", ['erkek', 'kadın'])
    Season = st.sidebar.selectbox("Mevsim", ['Summer', 'Winter'])
    Sport = st.sidebar.selectbox("Spor", ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Speed Skating',
       'Cross Country Skiing', 'Athletics', 'Ice Hockey', 'Swimming',
       'Badminton', 'Sailing', 'Biathlon', 'Gymnastics',
       'Art Competitions', 'Alpine Skiing', 'Handball', 'Weightlifting',
       'Wrestling', 'Luge', 'Water Polo', 'Hockey', 'Rowing', 'Bobsleigh',
       'Fencing', 'Equestrianism', 'Shooting', 'Boxing', 'Taekwondo',
       'Cycling', 'Diving', 'Canoeing', 'Tennis', 'Modern Pentathlon',
       'Figure Skating', 'Golf', 'Softball', 'Archery', 'Volleyball',
       'Synchronized Swimming', 'Table Tennis', 'Nordic Combined',
       'Baseball', 'Rhythmic Gymnastics', 'Freestyle Skiing',
       'Rugby Sevens', 'Trampolining', 'Beach Volleyball', 'Triathlon',
       'Ski Jumping', 'Curling', 'Snowboarding', 'Rugby',
       'Short Track Speed Skating', 'Skeleton', 'Lacrosse', 'Polo',
       'Cricket', 'Racquets', 'Motorboating', 'Military Ski Patrol',
       'Croquet', 'Jeu De Paume', 'Roque', 'Alpinism', 'Basque Pelota',
       'Aeronautics'])
    Year = st.sidebar.number_input("Yıl (Olimpiyat yılı)", min_value=1896, max_value=2024)
    Total_Games = st.sidebar.number_input("Toplam Katıldığı Olimpiyat Sayısı", min_value=1, max_value=10)
    Previously_Won_Gold = st.sidebar.selectbox("Daha Önce Altın Madalya Kazandı mı?", [0, 1])

    Yearly_Gold_Ratio = st.sidebar.number_input("Yıllık Altın Madalya Oranı (örn: 0.012)", min_value=0.0, max_value=0.2, value=0.01, format="%0.3f")
    Country_Gold_Ratio = st.sidebar.number_input("Ülke Altın Madalya Oranı (örn: 0.043)", min_value=0.0, max_value=0.2, value=0.01, format="%0.3f")

    Age_Group = 'Genç' if Age < 18 else 'Yetişkin' if Age < 30 else 'Olgun' if Age < 40 else 'Veteran'

    data = {
        'Age': Age,
        'Height': Height,
        'Weight': Weight,
        'Sex_Male': 1 if Sex == 'erkek' else 0,
        'Season_Winter': 1 if Season == 'Winter' else 0,
        'Sport_Athletics': 1 if Sport == 'Athletics' else 0,
        'Sport_Swimming': 1 if Sport == 'Swimming' else 0,
        'Sport_Gymnastics': 1 if Sport == 'Gymnastics' else 0,
        'Sport_Rowing': 1 if Sport == 'Rowing' else 0,
        'Sport_Football': 1 if Sport == 'Football' else 0,
        'Total_Games': Total_Games,
        'Previously_Won_Gold': Previously_Won_Gold,
        'Year': Year,
        'Yearly_Gold_Ratio': Yearly_Gold_Ratio,
        'Country_Gold_Ratio': Country_Gold_Ratio,
        'BMI': Weight / ((Height / 100) ** 2),
        'Age_Group_Olgun': 1 if Age_Group == 'Olgun' else 0,
        'Age_Group_Veteran': 1 if Age_Group == 'Veteran' else 0,
        'Age_Group_Yetişkin': 1 if Age_Group == 'Yetişkin' else 0,
    }
    return pd.DataFrame([data])

# Ana uygulama
st.title("Altın Madalya Tahmini Uygulaması")
st.markdown("Bir sporcunun altın madalya kazanma olasılığını tahmin etmek için bilgileri giriniz.")


model_option = st.selectbox("Model Seçin:", ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM'])
user_input_df = get_user_input()

features_to_scale = ['Age', 'Height', 'Weight', 'BMI', 'Total_Games', 'Year', 'Yearly_Gold_Ratio', 'Country_Gold_Ratio']

scaler = joblib.load("scaler.pkl")
user_input_df[features_to_scale] = scaler.transform(user_input_df[features_to_scale])

expected_features = joblib.load("feature_names.pkl")
for col in expected_features:
    if col not in user_input_df.columns:
        user_input_df[col] = 0
user_input_df = user_input_df[expected_features]

# model yüklemece
model_path = {
    'Logistic Regression': 'logistic_model.pkl',
    'Random Forest': 'rf_model.pkl',
    'XGBoost': 'xgb_model.pkl',
    'LightGBM': 'lgbm_model.pkl'
}

model = joblib.load(model_path[model_option])
prediction = model.predict(user_input_df)[0]
pred_proba = model.predict_proba(user_input_df)[0][1]

st.subheader("Tahmin Sonucu:")
if prediction == 1:
    st.success(f"Bu sporcunun altın madalya kazanma ihtimali YÜKSEK! (%{pred_proba * 100:.2f})")
else:
    st.warning(f"Bu sporcunun altın madalya kazanma ihtimali DÜŞÜK. (%{pred_proba * 100:.2f})")
