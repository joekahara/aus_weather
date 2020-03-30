# %% ------------ IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# %% ------------ GENERAL PRE-PROCESSING
# Read data from CSV dataset
df = pd.read_csv("weatherAUS.csv")

# Display df shape and sorted count of column values
print(df.shape, "\n")
print(df.count().sort_values(), "\n")

# Sunshine, Evaporation, Cloud3pm, Cloud9am all have > 54,000 missing values
# Drop them
features_to_drop = ['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am']
df.drop(features_to_drop, axis=1, inplace=True)

# Change date feature to datetime datatype
df.Date = pd.to_datetime(df.Date)

# Encode binary categorical features to one-hot encoding style values (i.e., 0 = 'no' and 1 = 'yes')
df.RainToday.replace(to_replace=['Yes', 'No'], value=['1', '0'], inplace=True)
df.RainTomorrow.replace(to_replace=['Yes', 'No'], value=['1', '0'], inplace=True)

# Encode remaining categorical features using one-hot encoding
categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
df_encoded = pd.get_dummies(df, prefix_sep="_",
                            columns=categorical_features)
# print(df_encoded.columns)

# %% ------------ INITIAL VISUALIZATION
# Display correlation matrix of features
sb.set(font_scale=0.75)
sb.heatmap(df.corr(), annot=True)

# Display histograms for numeric features
for col, values in df.items():
    if values.dtype != 'object':
        print("Plotting ", col)
        plt.hist(values, facecolor='peru', edgecolor='blue')
        plt.title(col)
        plt.show()

# %% ------------ SPECIFIC PRE-PROCESSING

# Split dataset into a number of unique locations
df_sydney = df_encoded[df_encoded['Location'] == 'Sydney'].reset_index(drop=True)
df_perth = df_encoded[df_encoded['Location'] == 'Perth'].reset_index(drop=True)
df_canberra = df_encoded[df_encoded['Location'] == 'Canberra'].reset_index(drop=True)
df_adelaide = df_encoded[df_encoded['Location'] == 'Adelaide'].reset_index(drop=True)
df_brisbane = df_encoded[df_encoded['Location'] == 'Brisbane'].reset_index(drop=True)

# Store all location-based datasets in a list for simple processing
dfs_by_location = [df_sydney, df_adelaide, df_brisbane, df_canberra, df_perth]

for i in range(len(dfs_by_location)):
    # Impute missing numerical values with k-NN
    imputer = KNNImputer()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_numerics = dfs_by_location[i].select_dtypes(include=numerics)
    df_imputed_numerics = pd.DataFrame(imputer.fit_transform(df_numerics), columns=df_numerics.columns)

    # Merge imputed numeric features with full data subset
    df_categoricals = dfs_by_location[i].drop(df_numerics.columns, axis=1)
    dfs_by_location[i] = pd.concat([df_categoricals, df_imputed_numerics], axis=1)
    # print(df_sydney.columns)
    print(dfs_by_location[i])

    # Split location dataframes into test/training sets based on a percentage
    dfs_by_location[i] = np.split(dfs_by_location[i], ([int(0.8*dfs_by_location[i].shape[0])]))
    print(dfs_by_location[i])

# %% ------------ TODO: FEATURE SELECTION

# %% ------------ MODEL DEVELOPMENT
rf_classifier = RandomForestClassifier()
rf_classifier.fit(dfs_by_location[0][0], dfs_by_location[0][1])
print(rf_classifier.feature_importances_)
# rf_prediction = rf_classifier.predict()
