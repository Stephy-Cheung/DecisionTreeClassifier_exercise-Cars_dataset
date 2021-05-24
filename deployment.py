import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import pickle 

from sklearn.tree import DecisionTreeClassifier 

def get_model():
    pickle_in = open('tree2.pkl', 'rb')
    treemodel = pickle.load(pickle_in)
    return treemodel

# load model
model = get_model()

st.title('Car Class predictor')

buying_cat = ['Low', 'Medium', 'High','Very high']
maint_cat = ['Low', 'Medium', 'High','Very high']
person_cat = ['2','4','More']
lug_boot_cat = ['Small','Medium', 'Big']
safety_cat = ['Low','Medium','High']

buying = st.selectbox('Buying price: ', buying_cat)
maint = st.selectbox('Maintainence: ', maint_cat)
door = st.slider('Number of doors: (choose \'5\' for 5 doors or above)', min_value=2,max_value=5,step =1)
person = st.selectbox('Number of people: ', person_cat )
lug_boot = st.selectbox('Luggage boot size: ', lug_boot_cat)
safety = st.selectbox('Safety performance: ', safety_cat)

buying = buying_cat.index(buying)
maint = maint_cat.index(maint)
person = person_cat.index(person)
lug_boot = lug_boot_cat.index(lug_boot)
safety = safety_cat.index(safety)

# Prediction
st.header('Prediction')

test_data = [[buying,maint,door,person,lug_boot,safety]]
prediction = model.predict(test_data)
st.text ('Car class range from : unacc, acc, good, very good')
if  prediction == 0: 
    st.subheader('This car belongs to class \'unacc\'')
elif prediction == 1: 
    st.subheader('This car belongs to class \'acc\'')
elif prediction == 2: 
    st.subheader('This car belongs to class \'good\'')
elif prediction == 3: 
    st.subheader('This car belongs to class \'very good\'')
