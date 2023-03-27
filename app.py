import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
#from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np 
import hashlib


# Define the hashing function
def hash_func(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % 10**8

train_data = pd.read_csv("/home/ahmedelheity/Downloads/Streamlit-master/data/train_data.csv")
training = train_data
training["protein_sequence"] = training["protein_sequence"].apply(hash_func)
training["data_source"] = training["data_source"].apply(hash_func)
features = training.drop(["seq_id","tm"], axis = 1)
target = training["tm"]
#x = np.array(data['YearsExperience']).reshape(-1,1)
#lr = LinearRegression()
#lr.fit(x,np.array(data['Salary']))
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# create a Gradient Boosting model and fit it to the training data
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42)
gb_model.fit(X_train, y_train)

st.title("Protein Stability Predictor")
st.image("data//1.jpeg",width = 800)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
if nav == "Home":
    
    if st.checkbox("Show Table"):
        st.table(training.head(20))
    
    graph = st.selectbox("What kind of Graph ? ",["PH"])

    val = st.slider("Filter data using years",7)
    data = train_data.loc[train_data["pH"]<= val]
    if graph == "PH":
        plt.figure(figsize = (10,5))
        plt.scatter(train_data["tm"],train_data["pH"])
        plt.ylim(0)
        plt.xlabel("pH")
        plt.ylabel("tm")
        plt.tight_layout()
        st.pyplot()
    
    
if nav == "Prediction":
    st.header("Measure the enzyme stability")
    val1 = st.text_input('Enter the Protein Sequence', '')
    val2 = st.number_input("Enter the  PH",0.00,10.00,step = 0.25)
    val3 = st.text_input('Enter the protein source', '')
    test_data = pd.DataFrame({
        'protein_sequence' : val1,
        'pH' : val2,
        'data_source' : val3,
        }, index=['row1'])
    test_data["protein_sequence"] = test_data["protein_sequence"].apply(hash_func)
    test_data["data_source"] = test_data["data_source"].apply(hash_func)
    #val = np.array(val).reshape(1,-1)
    pred =gb_model.predict(test_data)

    if st.button("Predict"):
        st.success(f"Your predicted tm is {pred}")

if nav == "Contribute":
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your Experience",0.0,20.0)
    sal = st.number_input("Enter your Salary",0.00,1000000.00,step = 1000.0)
    if st.button("submit"):
        to_add = {"YearsExperience":[ex],"Salary":[sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("data//Salary_Data.csv",mode='a',header = False,index= False)
        st.success("Submitted")
