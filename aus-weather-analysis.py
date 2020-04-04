# %% ------------ LIBRARY IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# %% ------------ FUNCTION DEFINITIONS

def preprocess_location(location_df):
    # Split location dataframes into test/training sets based on a percentage
    train, test = np.split(location_df, ([int(0.8 * location_df.shape[0])]))

    # Split test set by features and class to prepare for prediction and performance metrics
    test_class = test['RainTomorrow']
    test_features = test.drop('RainTomorrow', axis=1)

    # Impute missing values in training set and test set separately with k-NN
    imputer = KNNImputer()
    train = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)
    test_features = pd.DataFrame(imputer.fit_transform(test_features), columns=test_features.columns)

    # Split training set by features and class for use with sk-learn functions
    train_class = train['RainTomorrow']
    train_features = train.drop('RainTomorrow', axis=1)

    # TODO: CLASS BALANCING (under-sampling)
    # SMOTE over-sampling for 'RainTomorrow' class = 1 to balance classes in training set
    sm = SMOTE()
    train_features, train_class = sm.fit_resample(train_features, train_class)

    return train_features, train_class, test_features, test_class


def plot_confusion_matrix(cnf_matrix):
    class_names = [0, 1]
    # Configure plot and settings
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # Create and display heat-map
    sb.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion Matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def plot_roc_curve(roc):
    return


def print_metrics(location, model_type, cnf_matrix):
    print(location, model_type, " accuracy = ", (cnf_matrix[1][1] + cnf_matrix[0][0]) / cnf_matrix.sum().sum())
    print(location, model_type, " sensitivity = ", cnf_matrix[1][1] / (cnf_matrix[1][1] + cnf_matrix[1][0]))
    print(location, model_type, " specificity = ", cnf_matrix[0][0] / (cnf_matrix[0][0] + cnf_matrix[0][1]))
    print(location, model_type, " precision = ", cnf_matrix[1][1] / (cnf_matrix[1][1] + cnf_matrix[0][1]))


# %% ------------ GENERAL PRE-PROCESSING

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

# Read data from CSV dataset, sort by date and reset indices
df = pd.read_csv("weatherAUS.csv", parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

# Drop 'RISK_MM' column; it is data that reveals the class 'RainTomorrow'
df = df.drop('RISK_MM', axis=1)

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

# %% ------------ LOCATION-BASED DATASET PRE-PROCESSING

df_sydney_train_features, df_sydney_train_class, \
    df_sydney_test_features, df_sydney_test_class = preprocess_location(df_sydney)
df_perth_train_features, df_perth_train_class, \
    df_perth_test_features, df_perth_test_class = preprocess_location(df_perth)
df_canberra_train_features, df_canberra_train_class, \
    df_canberra_test_features, df_canberra_test_class = preprocess_location(df_canberra)
df_adelaide_train_features, df_adelaide_train_class, \
    df_adelaide_test_features, df_adelaide_test_class = preprocess_location(df_adelaide)
df_brisbane_train_features, df_brisbane_train_class, \
    df_brisbane_test_features, df_brisbane_test_class = preprocess_location(df_brisbane)

# %% ------------ TODO: FEATURE SELECTION


# %% ------------ RANDOM FOREST MODEL DEVELOPMENT

rf_sydney = RandomForestClassifier()
rf_sydney.fit(df_sydney_train_features, df_sydney_train_class)
df_sydney_test_pred_rf = rf_sydney.predict(df_sydney_test_features)

rf_perth = RandomForestClassifier()
rf_perth.fit(df_perth_train_features, df_perth_train_class)
df_perth_test_pred_rf = rf_perth.predict(df_perth_test_features)

rf_canberra = RandomForestClassifier()
rf_canberra.fit(df_canberra_train_features, df_canberra_train_class)
df_canberra_test_pred_rf = rf_canberra.predict(df_canberra_test_features)

rf_adelaide = RandomForestClassifier()
rf_adelaide.fit(df_adelaide_train_features, df_adelaide_train_class)
df_adelaide_test_pred_rf = rf_adelaide.predict(df_adelaide_test_features)

rf_brisbane = RandomForestClassifier()
rf_brisbane.fit(df_brisbane_train_features, df_brisbane_train_class)
df_brisbane_test_pred_rf = rf_brisbane.predict(df_brisbane_test_features)

# %% ------------ LINEAR REGRESSION MODEL DEVELOPMENT

lr_sydney = LogisticRegression()
lr_sydney.fit(df_sydney_train_features, df_sydney_train_class)
df_sydney_test_pred_lr = lr_sydney.predict(df_sydney_test_features)

lr_perth = LogisticRegression()
lr_perth.fit(df_perth_train_features, df_perth_train_class)
df_perth_test_pred_lr = lr_perth.predict(df_perth_test_features)

lr_canberra = LogisticRegression()
lr_canberra.fit(df_canberra_train_features, df_canberra_train_class)
df_canberra_test_pred_lr = lr_canberra.predict(df_canberra_test_features)

lr_adelaide = LogisticRegression()
lr_adelaide.fit(df_adelaide_train_features, df_adelaide_train_class)
df_adelaide_test_pred_lr = lr_adelaide.predict(df_adelaide_test_features)

lr_brisbane = LogisticRegression()
lr_brisbane.fit(df_brisbane_train_features, df_brisbane_train_class)
df_brisbane_test_pred_lr = lr_brisbane.predict(df_brisbane_test_features)

# %% ------------ k-NN LAZY LEARNING MODEL DEVELOPMENT

knn_sydney = KNeighborsClassifier()
knn_sydney.fit(df_sydney_train_features, df_sydney_train_class)
df_sydney_test_pred_knn = knn_sydney.predict(df_sydney_test_features)

knn_perth = KNeighborsClassifier()
knn_perth.fit(df_perth_train_features, df_perth_train_class)
df_perth_test_pred_knn = knn_perth.predict(df_perth_test_features)

knn_canberra = KNeighborsClassifier()
knn_canberra.fit(df_canberra_train_features, df_canberra_train_class)
df_canberra_test_pred_knn = knn_canberra.predict(df_canberra_test_features)

knn_adelaide = KNeighborsClassifier()
knn_adelaide.fit(df_adelaide_train_features, df_adelaide_train_class)
df_adelaide_test_pred_knn = knn_adelaide.predict(df_adelaide_test_features)

knn_brisbane = KNeighborsClassifier()
knn_brisbane.fit(df_brisbane_train_features, df_brisbane_train_class)
df_brisbane_test_pred_knn = knn_brisbane.predict(df_brisbane_test_features)

# %% ------------ SUPPORT VECTOR CLASSIFIER MODEL DEVELOPMENT

svm_sydney = SVC()
svm_sydney.fit(df_sydney_train_features, df_sydney_train_class)
df_sydney_test_pred_svm = svm_sydney.predict(df_sydney_test_features)

svm_perth = SVC()
svm_perth.fit(df_perth_train_features, df_perth_train_class)
df_perth_test_pred_svm = svm_perth.predict(df_perth_test_features)

svm_canberra = SVC()
svm_canberra.fit(df_canberra_train_features, df_canberra_train_class)
df_canberra_test_pred_svm = svm_canberra.predict(df_canberra_test_features)

svm_adelaide = SVC()
svm_adelaide.fit(df_adelaide_train_features, df_adelaide_train_class)
df_adelaide_test_pred_svm = svm_adelaide.predict(df_adelaide_test_features)

svm_brisbane = SVC()
svm_brisbane.fit(df_brisbane_train_features, df_brisbane_train_class)
df_brisbane_test_pred_svm = svm_brisbane.predict(df_brisbane_test_features)
