import pandas as pd 
import streamlit as st
import numpy as np
import sklearn
import pickle
st.title('SALES PREDICTION')
st.text('Welcome')
st.image('immage.png.webp', width= 300)
dataa = pd.read_csv(r'Advertising Budget and Sales.csv')
corr = pd.read_csv(r'corrr.csv')
Modeltest = pd.read_csv(r'advtest.csv')


st.subheader('The data use for this model')
st.write(dataa)
st.subheader('Data Relationship')
st.write(corr)

import streamlit as st
import pandas as pd
import seaborn as sns




st.sidebar.title('Make your prediction')
st.sidebar.image('te.png',width= 40)
file = open('advmodell.pkl', 'rb')  
model = pickle.load(file) 

def main():
    st.sidebar.title('Model')

   
    tv_ad_budget = st.sidebar.text_input('TV Ad Budget ($)')
    radio_ad_budget = st.sidebar.text_input('Radio Ad Budget ($)')
    newspaper_ad_budget = st.sidebar.text_input('Newspaper Ad Budget ($)')

    if st.sidebar.button('Predict'):
        
        tv_ad_budget = float(tv_ad_budget)
        radio_ad_budget = float(radio_ad_budget)
        newspaper_ad_budget = float(newspaper_ad_budget)

       
        prediction = model.predict([[tv_ad_budget, radio_ad_budget, newspaper_ad_budget]])
        output = round(prediction[0], 2)

        
        st.sidebar.success(f'Your estimated sales would be ${output}')

if __name__ == '__main__':
    main()