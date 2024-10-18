import streamlit as st
import pandas as pd 
import plotly.express as px 
import base64
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder


### defined functions 
@st.cache_data
def load_arch():
    model= pickle.load(open("svc_heartattack_classifier.pkl", 'rb'))
    df=pd.read_csv('heart.csv')
    df.rename(columns={
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest_Pain_Type',
    'trtbps': 'Resting_Blood_Pressure',
    'chol': 'Cholesterol',
    'fbs': 'Fasting_Blood_Sugar',
    'restecg': 'Resting_ECG',
    'thalachh': 'Max_Heart_Rate_Achieved',
    'exng': 'Exercise_Induced_Angina',
    'oldpeak': 'ST_Depression',
    'slp': 'Slope_of_ST_Segment',
    'caa': 'Number_of_Major_Vessels',
    'thall': 'Thalassemia'
    }, inplace=True)
    df.drop_duplicates(inplace=True)
    encoder=OneHotEncoder(sparse_output=False).fit(df[['Sex',
 'Chest_Pain_Type',
 'Resting_ECG',
 'Exercise_Induced_Angina',
 'Slope_of_ST_Segment',
 'Number_of_Major_Vessels',
 'Thalassemia']])
    scaler=StandardScaler().fit(df[['Age', 'Resting_Blood_Pressure', 'Cholesterol', 'Max_Heart_Rate_Achieved', 'ST_Depression']])
    return model,encoder,scaler

def predict_attack(data,encoder,scaler,model):
    df1=pd.DataFrame(data,columns=['Age', 'Sex', 'Chest_Pain_Type', 'Resting_Blood_Pressure',
       'Cholesterol', 'Resting_ECG',
       'Max_Heart_Rate_Achieved', 'Exercise_Induced_Angina', 'ST_Depression',
       'Slope_of_ST_Segment', 'Number_of_Major_Vessels', 'Thalassemia'])
    cat_c=['Sex',
    'Chest_Pain_Type',
    'Resting_ECG',
    'Exercise_Induced_Angina',
    'Slope_of_ST_Segment',
    'Number_of_Major_Vessels',
    'Thalassemia']
    numerical_columns=['Age', 
    'Resting_Blood_Pressure', 
    'Cholesterol', 
    'Max_Heart_Rate_Achieved', 
    'ST_Depression']
    encoded = encoder.transform(df1[cat_c])
    df1.drop(cat_c,axis=1,inplace=True)
    df1[encoder.get_feature_names_out(cat_c)] = encoded
    df1[numerical_columns] = scaler.transform(df1[numerical_columns])
    prediction=model.predict_proba(df1)
    prediction_df=pd.DataFrame(prediction[0],columns=['value'])
    prediction_df['class']=['No Heart Attack','Heart Attack']
    return prediction_df


### defining variables
#styling
page_style=f"""
<style>
[data-testid="stAppViewContainer"]{{
    background-color:#e3647f;
}}

[data-testid="stSidebar"]>div:first-child{{
    background-color: #03071a;
}}

</style>
"""

#features

genders=['Female','Male']
chest_pain_types=['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic']
number_of_major_vesseles=['1','2','3','4','5']
resting_ECG_types=['Normal','having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)','showing probable or definite left ventricular hypertrophy by Estes criteria']
angina_exs=['No','Yes']
slps=['1','2','3']
thalls=['1','2','3','4']

#learning
model,encoder,scaler=load_arch()

# result design

images=['case1.png','case2.png','case3.png','case4.png']
captions=['Great!!....your health looks amazing. keep going....','Nice!!...do not forget to care about your health...',
'OH!!....you have to care about your heath...','mmm!!!....you need to visit the doctor as soon as possible.']

###dashboard  

#main board
st.markdown(page_style,unsafe_allow_html=True)
st.title("Heart Attack Classifier")
st.caption("**_This board has been developed to help you_ :blue[avoiding sudden heart attack] _wishing you all to have a good health and to live well._**")

#features values
st.sidebar.header("Your midical data record")
sidebar_c1,sidebar_c2=st.sidebar.columns(2)

with sidebar_c1:
    age=st.text_input("Age:")
    resting_blood_pressure=st.text_input("trtbps:")
    chest_pain=st.selectbox("Chest pain:",chest_pain_types,index=None,placeholder="...")
    maximum_heartrate_achieved=st.text_input("thalach:")
    st_depresion=st.text_input("Old peak:")
    vcc=st.selectbox("n(major vesseles):",number_of_major_vesseles,index=None,placeholder="...")
with sidebar_c2:
    sex=st.selectbox("Gender:",genders,index=None,placeholder="...")
    cholestrole=st.text_input("Cholestrole:")
    resting_ecg=st.selectbox("Resting ECG:",resting_ECG_types,index=None,placeholder="...")
    exercise_include_angina=st.selectbox("exang:",angina_exs,index=None,placeholder="...")
    slope_st_segment=st.selectbox("slope:",slps,index=None,placeholder="...")
    thall=st.selectbox("Thalium Stress:",thalls,index=None,placeholder="...")
if st.sidebar.button("Check",type="primary",use_container_width=True):
   record=[[
            int(age),
            genders.index(sex),
            chest_pain_types.index(chest_pain),
            int(resting_blood_pressure),
            int(cholestrole),
            resting_ECG_types.index(resting_ecg),
            int(maximum_heartrate_achieved),
            angina_exs.index(exercise_include_angina),
            np.float64(st_depresion),
            slps.index(slope_st_segment),
            number_of_major_vesseles.index(vcc),
            thalls.index(thall)
        ]]
   res=predict_attack(record,encoder,scaler,model)
   attack_val=res['value'].values[1]*100
   con=st.container(height=350,border=True)
   c1,c2=con.columns(2)
   c2.plotly_chart(px.pie(res,values='value',names='class'))
   if attack_val<=10:
        c1.image(images[0])
        st.caption(captions[0])
   elif attack_val>10 and attack_val<=40:
        c1.image(images[1])
        st.caption(captions[1])
   elif attack_val>40 and attack_val<=70:
        c1.image(images[2])
        st.caption(captions[2])
   else:
        c1.image(images[3])
        st.caption(captions[3])