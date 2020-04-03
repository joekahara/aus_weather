# %% ------------ IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# %% ------------ GENERAL PRE-PROCESSING

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

# Read data from CSV dataset, sort by date and reset indices
df = pd.read_csv("weatherAUS.csv", parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

# Drop 'RISK_MM' column; it is data that reveals the class 'RainTomorrow'
df = df.drop('RISK_MM', axis=1)

# Display dataframe shape and sorted count of column values
print(df.shape, "\n")
print(df.count().sort_values(), "\n")

# Sunshine, Evaporation, Cloud3pm, Cloud9am all have > 54,000 missing values
# Drop these columns and Date as we no longer need it
features_to_drop = ['Date', 'Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am']
df.drop(features_to_drop, axis=1, inplace=True)

# Encode binary categorical features to one-hot encoding style values (i.e., 0 = 'no' and 1 = 'yes')
df['RainToday'].replace(to_replace=['Yes', 'No'], value=['1', '0'], inplace=True)
df['RainTomorrow'].replace(to_replace=['Yes', 'No'], value=['1', '0'], inplace=True)

# Change remaining 'object' datatypes to numeric
df['RainToday'] = pd.to_numeric(df['RainToday'])
df['RainTomorrow'] = pd.to_numeric(df['RainTomorrow'])

# Encode remaining categorical features using one-hot encoding
categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
df_encoded = pd.get_dummies(df, prefix_sep="_",
                            columns=categorical_features)

print(df_encoded.shape)

'''# %% ------------ INITIAL VISUALIZATION
# Display correlation matrix of features
sb.set(font_scale=0.75)
sb.heatmap(df.corr(), annot=True)

# Display histograms for numeric features
for col, values in df.items():
    if values.dtype != 'object':
        print("Plotting ", col)
        plt.hist(values, facecolor='peru', edgecolor='blue')
        plt.title(col)
        plt.show()'''

# %% ------------ CREATION OF LOCATION-BASED DATASETS

# Split dataset into a number of unique locations
# Drop location feature as it's no longer needed
df_sydney = df_encoded[df_encoded['Location'] == 'Sydney'].drop(['Location'], axis=1).reset_index(drop=True)
df_perth = df_encoded[df_encoded['Location'] == 'Perth'].drop(['Location'], axis=1).reset_index(drop=True)
df_canberra = df_encoded[df_encoded['Location'] == 'Canberra'].drop(['Location'], axis=1).reset_index(drop=True)
df_adelaide = df_encoded[df_encoded['Location'] == 'Adelaide'].drop(['Location'], axis=1).reset_index(drop=True)
df_brisbane = df_encoded[df_encoded['Location'] == 'Brisbane'].drop(['Location'], axis=1).reset_index(drop=True)

# %% ------------ SYDNEY DATASET PRE-PROCESSING

print(df_sydney.isna().sum().sum())
print(df_sydney.count().sum().sum())

# Split location dataframes into test/training sets based on a percentage
df_sydney_train, df_sydney_test = np.split(df_sydney, ([int(0.8*df_sydney.shape[0])]))

# Split test set by features and class to prepare for prediction and performance metrics
df_sydney_test_class = df_sydney_test['RainTomorrow']
df_sydney_test_features = df_sydney_test.drop('RainTomorrow', axis=1)

# Impute missing values in training set and test set separately with k-NN
imputer = KNNImputer()
df_sydney_train = pd.DataFrame(imputer.fit_transform(df_sydney_train), columns=df_sydney_train.columns)
df_sydney_test_features = pd.DataFrame(imputer.fit_transform(df_sydney_test_features),
                                       columns=df_sydney_test_features.columns)

# Split training set by features and class for use with sk-learn functions
df_sydney_train_class = df_sydney_train['RainTomorrow']
df_sydney_train_features = df_sydney_train.drop('RainTomorrow', axis=1)

# SMOTE oversampling for 'RainTomorrow' class = 1 to balance classes in training set
sm = SMOTE()
print(df_sydney_train_class.shape)
df_sydney_train_features, df_sydney_train_class = sm.fit_resample(df_sydney_train_features, df_sydney_train_class)
print(df_sydney_train_class.shape)

# %% ------------ TODO: FEATURE SELECTION


# %% ------------ TODO: CLASS BALANCING


# %% ------------ RANDOM FOREST MODEL DEVELOPMENT

rf_classifier = RandomForestClassifier()
rf_classifier.fit(df_sydney_train_features, df_sydney_train_class)
df_sydney_test_pred = rf_classifier.predict(df_sydney_test_features)

print("RF Classifier accuracy = ", accuracy_score(df_sydney_test_class, df_sydney_test_pred))
cm = confusion_matrix(df_sydney_test_class, df_sydney_test_pred)
print(cm)
pd.crosstab(df_sydney_test_class, df_sydney_test_pred, rownames=['True'], colnames=['Predicted'], margins=True)

sb.heatmap(cm)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
