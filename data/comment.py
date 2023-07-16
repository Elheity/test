'''
train_data = pd.read_csv("/home/ahmedelheity/Downloads/Streamlit-master/data/train_data.csv")
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


# Split the data into train and validation set
training_new, rem_df = train_test_split(training, train_size=0.8,random_state=99)

val_df, test_df = train_test_split(rem_df, test_size=0.5,random_state=123)

X_train = training_new.drop(columns=['tm'])
y_train = training_new['tm']

X_val = val_df.drop(columns=['tm'])
y_val = val_df['tm']

X_test = test_df.drop(columns=['tm'])
y_test = test_df['tm']

#features = training.drop(["seq_id","tm"], axis = 1)
#target = training["tm"]
#x = np.array(data['YearsExperience']).reshape(-1,1)
#lr = LinearRegression()
#lr.fit(x,np.array(data['Salary']))
#X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# create a Gradient Boosting model and fit it to the training data
#gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42)
#gb_model.fit(X_train, y_train)
#XGBoost Model
model = xgb.XGBRegressor(learning_rate=0.1, max_depth=8, n_estimators=200, tree_method="hist",random_state=98)
model.fit(X_train, y_train)

rfecv = RFECV(estimator= model, step = 5, cv = 5, scoring='neg_mean_squared_error')
rfecv = rfecv.fit(X_train, y_train)

print("The optimal number of features:", rfecv.n_features_)
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

best_features = list(X_train.columns[rfecv.support_])

X_train_new = X_train[best_features]
X_val_new = X_val[best_features]
X_test_new = X_test[best_features]

model1 = xgb.XGBRegressor(learning_rate=0.1, max_depth=20, n_estimators=250, tree_method="hist",random_state=123)
model1.fit(X_train_new, y_train)




'''