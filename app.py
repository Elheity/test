import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
#from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np 
import hashlib
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re
import plotly.express as px

# PhysioChemical Properties of Amino acids

#Aromaticity
def calculate_aromaticity(row):
  sequence = str(row[1])
  X = ProteinAnalysis(sequence)
  return "%0.2f" % X.aromaticity()

#Molecular Weight
def calculate_molecular_weight(row):
  sequence = str(row[1])
  X = ProteinAnalysis(sequence)
  return "%0.2f" % X.molecular_weight()

#Instability Index
def calculate_instability_index(row):
  sequence = str(row[1])
  X = ProteinAnalysis(sequence)
  return "%0.2f" % X.instability_index()

#Hydrophobicity
def calculate_hydrophobicity(row):
  sequence = str(row[1])
  X = ProteinAnalysis(sequence)
  return "%0.2f" % X.gravy()

#Isoelectric Point
def calculate_isoelectric_point(row):
  sequence = str(row[1])
  X = ProteinAnalysis(sequence)
  return "%0.2f" % X.isoelectric_point()

#Charge
def calculate_charge(row):
  sequence = str(row[1])
  X = ProteinAnalysis(sequence)
  return "%0.2f" % X.charge_at_pH(row[2])

#####################
# Define the feature Engineering for test data function
################
def calculate_aromaticity_test(row):
  sequence = str(row[0])
  X = ProteinAnalysis(sequence)
  length = X.length
  if length > 0:
    return "%0.2f" % X.aromaticity()
  else:
    return "%0.2f" % 0.0

#Molecular Weight
def calculate_molecular_weight_test(row):
  sequence = str(row[0])
  X = ProteinAnalysis(sequence)
  length = X.length
  if length > 0:
    return "%0.2f" % X.molecular_weight()
  else:
    return "%0.2f" % 0.0

#Instability Index
def calculate_instability_index_test(row):
  sequence = str(row[0])
  X = ProteinAnalysis(sequence)
  length = X.length
  if length > 0:
    return "%0.2f" % X.instability_index()
  else:
    return "%0.2f" % 0.0

#Hydrophobicity
def calculate_hydrophobicity_test(row):
  sequence = str(row[0])
  X = ProteinAnalysis(sequence)
  length = X.length
  if length > 0:
    return "%0.2f" % X.gravy()
  else:
    return "%0.2f" % 0.0

#Isoelectric Point
def calculate_isoelectric_point_test(row):
  sequence = str(row[0])
  X = ProteinAnalysis(sequence)
  length = X.length
  if length > 0:
    return "%0.2f" % X.isoelectric_point()
  else:
    return "%0.2f" % 0.0

#Charge
def calculate_charge_test(row):
  sequence = str(row[0])
  X = ProteinAnalysis(sequence)
  length = X.length
  if length > 0:
    return "%0.2f" % X.charge_at_pH(row[1])
  else:
    return "%0.2f" % 0.0
# Define the hashing function
def hash_func(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % 10**8


train_data = pd.read_csv("data/train_data.csv")
training = train_data
training=training.dropna(how='all')
training['pH'] = training['pH'].fillna(training['pH'].mean())
training.drop_duplicates(subset=['protein_sequence','pH','data_source'],inplace=True)
training = training.drop(['data_source'],axis=1)

amino_count = training['protein_sequence'].str.split('').explode().value_counts().drop('')

# Protein Sequence Length as a column
training["protein_length"] = training["protein_sequence"].apply(lambda x: len(x))

def return_amino_acid_df(df):
  # Feature Engineering on Train Data
  amino_acids=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
  for amino_acid in amino_acids:
    df[amino_acid]=df['protein_sequence'].str.count(amino_acid,re.I)/df['protein_length']
    #df[amino_acid]=df['protein_sequence'].str.count(amino_acid,re.I)
  return df

training = return_amino_acid_df(training)

#training["protein_sequence"] = training["protein_sequence"].apply(hash_func)
#training["data_source"] = training["data_source"].apply(hash_func)
training['Aromaticity'] = training.apply(calculate_aromaticity, axis=1)
training['Molecular Weight'] = training.apply(calculate_molecular_weight, axis=1)
training['Instability Index'] = training.apply(calculate_instability_index, axis=1)
training['Hydrophobicity'] = training.apply(calculate_hydrophobicity, axis=1)
training['Isoelectric Point'] = training.apply(calculate_isoelectric_point, axis=1)
training['Charge'] = training.apply(calculate_charge, axis=1)

training.drop(columns=["protein_length"], inplace=True)
training.drop(columns=["protein_sequence", "seq_id"], inplace=True)

# Reset the DataFrame indexes
training.reset_index(drop=True, inplace=True)


training['Aromaticity'] = pd.to_numeric(training['Aromaticity'])
training['Molecular Weight'] = pd.to_numeric(training['Molecular Weight'])
training['Instability Index'] = pd.to_numeric(training['Instability Index'])
training['Hydrophobicity'] = pd.to_numeric(training['Hydrophobicity'])
training['Isoelectric Point'] = pd.to_numeric(training['Isoelectric Point'])
training['Charge'] = pd.to_numeric(training['Charge'])

features = training.drop(["tm"], axis = 1)
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
st.set_option('deprecation.showPyplotGlobalUse', False)
nav = st.sidebar.radio("Navigation",["Home","Prediction","About"])
if nav == "Home":
    
    if st.checkbox("Show Table"):
        st.table(training.head(20))
    
    graph = st.selectbox("What kind of Graph ? ",["Amino acid count", "tm"])

    #val = st.slider("Filter data using years",7)
    #data = train_data.loc[train_data["pH"]<= val]
    if graph == "tm":
        #plt.figure(figsize = (10,5))
        #plt.scatter(train_data["tm"],train_data["pH"])
        #plt.ylim(0)
        #plt.xlabel("pH")
        #plt.ylabel("tm")
        #plt.tight_layout()
        #st.pyplot()
        fig= px.histogram(training, x='tm', title="tm Distribution", template='plotly_dark',width=800,height=400)
        st.plotly_chart(fig)
    if graph == "Amino acid count":
        fig = px.bar(amino_count, x=amino_count.index, y='count', color=amino_count.index)
        fig.update_layout(
            title='Amino Acid Count',
            height=600,
            template='ggplot2'
        )
        st.plotly_chart(fig)
    
    
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
    
    test_data = test_data.drop(['data_source'],axis=1)

    amino_count = test_data['protein_sequence'].str.split('').explode().value_counts().drop('')

    # Protein Sequence Length as a column
    test_data["protein_length"] = test_data["protein_sequence"].apply(lambda x: len(x))

          
    if not test_data.empty:
      test_data = return_amino_acid_df(test_data)
            #print("shape", test_data.shape)
            #test_data["protein_sequence"] = test_data["protein_sequence"].apply(hash_func)
            #test_data["data_source"] = test_data["data_source"].apply(hash_func)
      test_data['Aromaticity'] = test_data.apply(calculate_aromaticity_test, axis=1)
      test_data['Molecular Weight'] = test_data.apply(calculate_molecular_weight_test, axis=1)
      test_data['Instability Index'] = test_data.apply(calculate_instability_index_test, axis=1)
      test_data['Hydrophobicity'] = test_data.apply(calculate_hydrophobicity_test, axis=1)
      test_data['Isoelectric Point'] = test_data.apply(calculate_isoelectric_point_test, axis=1)
      test_data['Charge'] = test_data.apply(calculate_charge_test, axis=1)
  
      test_data.drop(columns=["protein_length"], inplace=True)
      test_data.drop(columns=["protein_sequence"], inplace=True)
  
            # Reset the DataFrame indexes
      test_data.reset_index(drop=True, inplace=True)
  
  
      test_data['Aromaticity'] = pd.to_numeric(test_data['Aromaticity'])
      test_data['Molecular Weight'] = pd.to_numeric(test_data['Molecular Weight'])
      test_data['Instability Index'] = pd.to_numeric(test_data['Instability Index'])
      test_data['Hydrophobicity'] = pd.to_numeric(test_data['Hydrophobicity'])
      test_data['Isoelectric Point'] = pd.to_numeric(test_data['Isoelectric Point'])
      test_data['Charge'] = pd.to_numeric(test_data['Charge'])
  
            #Bestfeatures= 
            #test_data["protein_sequence"] = test_data["protein_sequence"].apply(hash_func)
            #test_data["data_source"] = test_data["data_source"].apply(hash_func)
            #val = np.array(val).reshape(1,-1)
      if st.checkbox("Show Table"):
          st.table(test_data.head(20))
      if test_data.isna():
        st.text("Please, Enter the data")
      #val = np.array(val).reshape(1,-1)
      else:
        pred =gb_model.predict(test_data)
  
        if st.button("Predict"):
            st.success(f"The percentage of stability is  {pred}")

if nav == "About":
    st.title("About our model")
    st.text("The process of understanding the effect of pH on proteins ")
    st.text("and its potential contribution to mutations")
