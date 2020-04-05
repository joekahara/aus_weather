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
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# %% ------------ FUNCTION DEFINITIONS

def preprocess_location(location_df, class_balancing_type, feature_selection):
    # Split location dataframes into test/training sets based on a percentage
    train, test = np.split(location_df, ([int(0.3 * location_df.shape[0])]))

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

    if class_balancing_type == 'undersampling':
        # Random under-sampling for 'RainTomorrow' class = 1 to balance classes in training set
        rand_undersampler = RandomUnderSampler()
        train_features, train_class = rand_undersampler.fit_resample(train_features, train_class)

    if class_balancing_type == 'oversampling':
        # SMOTE over-sampling for 'RainTomorrow' class = 1 to balance classes in training set
        smote_oversampler = SMOTE()
        train_features, train_class = smote_oversampler.fit_resample(train_features, train_class)

    #if feature_selection:

    return train_features, train_class, test_features, test_class


def plot_confusion_matrix(cnf_matrix, location, model_type):
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
    plt.title((location + " " + model_type), y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def plot_roc_curve(actual, pred_rf, pred_lr, pred_knn, pred_svm):
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(actual, pred_rf)
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(actual, pred_lr)
    fpr_knn, tpr_knn, thresholds_knn = roc_curve(actual, pred_knn)
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(actual, pred_svm)
    
    plt.plot(fpr_rf, tpr_rf, 'r-', label='RF')
    plt.plot(fpr_lr, tpr_lr, 'b-', label='LR')
    plt.plot(fpr_knn, tpr_knn, 'g-', label='KNN')
    plt.plot(fpr_svm, tpr_svm, 'c-', label='SVM')
    plt.plot([0, 1], [0, 1], 'k-', label='random')
    plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'k-', label='perfect')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


def print_auc(location, actual, pred_rf, pred_lr, pred_knn, pred_svm):
    auc_rf = roc_auc_score(actual, pred_rf)
    auc_lr = roc_auc_score(actual, pred_lr)
    auc_knn = roc_auc_score(actual, pred_knn)
    auc_svm = roc_auc_score(actual, pred_svm)
    print(location, 'AUC RF = ', auc_rf)
    print(location, 'AUC LR = ', auc_lr)
    print(location, 'AUC KNN = ', auc_knn)
    print(location, 'AUC SVM = ', auc_svm, '\n')


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

# Change remaining 'object' data-types to numeric
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

df_sydney_train_features, df_sydney_train_class, df_sydney_test_features, \
    df_sydney_test_class = preprocess_location(df_sydney, class_balancing_type='oversampling', feature_selection=True)
df_perth_train_features, df_perth_train_class, df_perth_test_features, \
    df_perth_test_class = preprocess_location(df_perth, class_balancing_type='oversampling', feature_selection=True)
df_canberra_train_features, df_canberra_train_class, df_canberra_test_features, \
    df_canberra_test_class = preprocess_location(df_canberra, class_balancing_type='oversampling', feature_selection=True)
df_adelaide_train_features, df_adelaide_train_class, df_adelaide_test_features, \
    df_adelaide_test_class = preprocess_location(df_adelaide, class_balancing_type='oversampling', feature_selection=True)
df_brisbane_train_features, df_brisbane_train_class, df_brisbane_test_features, \
    df_brisbane_test_class = preprocess_location(df_brisbane, class_balancing_type='oversampling', feature_selection=True)

# %% ------------ TODO: FEATURE SELECTION


# %% ------------ RANDOM FOREST MODEL AND METRICS

rf_sydney = RandomForestClassifier()
rf_sydney.fit(df_sydney_train_features, df_sydney_train_class)
df_sydney_test_pred_rf = rf_sydney.predict(df_sydney_test_features)

cm = confusion_matrix(y_pred=df_sydney_test_pred_rf, y_true=df_sydney_test_class)
plot_confusion_matrix(cm, location='Sydney', model_type='random forest')

rf_perth = RandomForestClassifier()
rf_perth.fit(df_perth_train_features, df_perth_train_class)
df_perth_test_pred_rf = rf_perth.predict(df_perth_test_features)

cm = confusion_matrix(y_pred=df_perth_test_pred_rf, y_true=df_perth_test_class)
plot_confusion_matrix(cm, location='Perth', model_type='random forest')

rf_canberra = RandomForestClassifier()
rf_canberra.fit(df_canberra_train_features, df_canberra_train_class)
df_canberra_test_pred_rf = rf_canberra.predict(df_canberra_test_features)

cm = confusion_matrix(y_pred=df_canberra_test_pred_rf, y_true=df_canberra_test_class)
plot_confusion_matrix(cm, location='Canberra', model_type='random forest')

rf_adelaide = RandomForestClassifier()
rf_adelaide.fit(df_adelaide_train_features, df_adelaide_train_class)
df_adelaide_test_pred_rf = rf_adelaide.predict(df_adelaide_test_features)

cm = confusion_matrix(y_pred=df_adelaide_test_pred_rf, y_true=df_adelaide_test_class)
plot_confusion_matrix(cm, location='Adelaide', model_type='random forest')

rf_brisbane = RandomForestClassifier()
rf_brisbane.fit(df_brisbane_train_features, df_brisbane_train_class)
df_brisbane_test_pred_rf = rf_brisbane.predict(df_brisbane_test_features)

cm = confusion_matrix(y_pred=df_brisbane_test_pred_rf, y_true=df_brisbane_test_class)
plot_confusion_matrix(cm, location='Brisbane', model_type='random forest')

# %% ------------ LINEAR REGRESSION MODEL AND METRICS

lr_sydney = LogisticRegression(max_iter=5000)
lr_sydney.fit(df_sydney_train_features, df_sydney_train_class)
df_sydney_test_pred_lr = lr_sydney.predict(df_sydney_test_features)

cm = confusion_matrix(y_pred=df_sydney_test_pred_lr, y_true=df_sydney_test_class)
plot_confusion_matrix(cm, location='Sydney', model_type='Logistic Regression')

lr_perth = LogisticRegression(max_iter=5000)
lr_perth.fit(df_perth_train_features, df_perth_train_class)
df_perth_test_pred_lr = lr_perth.predict(df_perth_test_features)

cm = confusion_matrix(y_pred=df_perth_test_pred_lr, y_true=df_perth_test_class)
plot_confusion_matrix(cm, location='Perth', model_type='Logistic Regression')

lr_canberra = LogisticRegression(max_iter=5000)
lr_canberra.fit(df_canberra_train_features, df_canberra_train_class)
df_canberra_test_pred_lr = lr_canberra.predict(df_canberra_test_features)

cm = confusion_matrix(y_pred=df_canberra_test_pred_lr, y_true=df_canberra_test_class)
plot_confusion_matrix(cm, location='Canberra', model_type='Logistic Regression')

lr_adelaide = LogisticRegression(max_iter=5000)
lr_adelaide.fit(df_adelaide_train_features, df_adelaide_train_class)
df_adelaide_test_pred_lr = lr_adelaide.predict(df_adelaide_test_features)

cm = confusion_matrix(y_pred=df_adelaide_test_pred_lr, y_true=df_adelaide_test_class)
plot_confusion_matrix(cm, location='Adelaide', model_type='Logistic Regression')

lr_brisbane = LogisticRegression(max_iter=5000)
lr_brisbane.fit(df_brisbane_train_features, df_brisbane_train_class)
df_brisbane_test_pred_lr = lr_brisbane.predict(df_brisbane_test_features)

cm = confusion_matrix(y_pred=df_brisbane_test_pred_lr, y_true=df_brisbane_test_class)
plot_confusion_matrix(cm, location='Brisbane', model_type='Logistic Regression')

# %% ------------ k-NN LAZY LEARNING MODEL AND METRICS

knn_sydney = KNeighborsClassifier()
knn_sydney.fit(df_sydney_train_features, df_sydney_train_class)
df_sydney_test_pred_knn = knn_sydney.predict(df_sydney_test_features)

cm = confusion_matrix(y_pred=df_sydney_test_pred_knn, y_true=df_sydney_test_class)
plot_confusion_matrix(cm, location='Sydney', model_type='k-NN Classifier')

knn_perth = KNeighborsClassifier()
knn_perth.fit(df_perth_train_features, df_perth_train_class)
df_perth_test_pred_knn = knn_perth.predict(df_perth_test_features)

cm = confusion_matrix(y_pred=df_perth_test_pred_knn, y_true=df_perth_test_class)
plot_confusion_matrix(cm, location='Perth', model_type='k-NN Classifier')

knn_canberra = KNeighborsClassifier()
knn_canberra.fit(df_canberra_train_features, df_canberra_train_class)
df_canberra_test_pred_knn = knn_canberra.predict(df_canberra_test_features)

cm = confusion_matrix(y_pred=df_canberra_test_pred_knn, y_true=df_canberra_test_class)
plot_confusion_matrix(cm, location='Canberra', model_type='k-NN Classifier')

knn_adelaide = KNeighborsClassifier()
knn_adelaide.fit(df_adelaide_train_features, df_adelaide_train_class)
df_adelaide_test_pred_knn = knn_adelaide.predict(df_adelaide_test_features)

cm = confusion_matrix(y_pred=df_adelaide_test_pred_knn, y_true=df_adelaide_test_class)
plot_confusion_matrix(cm, location='Adelaide', model_type='k-NN Classifier')

knn_brisbane = KNeighborsClassifier()
knn_brisbane.fit(df_brisbane_train_features, df_brisbane_train_class)
df_brisbane_test_pred_knn = knn_brisbane.predict(df_brisbane_test_features)

cm = confusion_matrix(y_pred=df_brisbane_test_pred_knn, y_true=df_brisbane_test_class)
plot_confusion_matrix(cm, location='Brisbane', model_type='k-NN Classifier')

# %% ------------ SUPPORT VECTOR CLASSIFIER MODEL AND METRICS

svm_sydney = SVC(max_iter=5000)
svm_sydney.fit(df_sydney_train_features, df_sydney_train_class)
df_sydney_test_pred_svm = svm_sydney.predict(df_sydney_test_features)

cm = confusion_matrix(y_pred=df_sydney_test_pred_svm, y_true=df_sydney_test_class)
plot_confusion_matrix(cm, location='Sydney', model_type='Support Vector Machine')

svm_perth = SVC(max_iter=5000)
svm_perth.fit(df_perth_train_features, df_perth_train_class)
df_perth_test_pred_svm = svm_perth.predict(df_perth_test_features)

cm = confusion_matrix(y_pred=df_perth_test_pred_svm, y_true=df_perth_test_class)
plot_confusion_matrix(cm, location='Perth', model_type='Support Vector Machine')

svm_canberra = SVC(max_iter=5000)
svm_canberra.fit(df_canberra_train_features, df_canberra_train_class)
df_canberra_test_pred_svm = svm_canberra.predict(df_canberra_test_features)

cm = confusion_matrix(y_pred=df_canberra_test_pred_svm, y_true=df_canberra_test_class)
plot_confusion_matrix(cm, location='Canberra', model_type='Support Vector Machine')

svm_adelaide = SVC(max_iter=5000)
svm_adelaide.fit(df_adelaide_train_features, df_adelaide_train_class)
df_adelaide_test_pred_svm = svm_adelaide.predict(df_adelaide_test_features)

cm = confusion_matrix(y_pred=df_adelaide_test_pred_svm, y_true=df_adelaide_test_class)
plot_confusion_matrix(cm, location='Adelaide', model_type='Support Vector Machine')

svm_brisbane = SVC(max_iter=5000)
svm_brisbane.fit(df_brisbane_train_features, df_brisbane_train_class)
df_brisbane_test_pred_svm = svm_brisbane.predict(df_brisbane_test_features)

cm = confusion_matrix(y_pred=df_brisbane_test_pred_svm, y_true=df_brisbane_test_class)
plot_confusion_matrix(cm, location='Brisbane', model_type='Support Vector Machine')

# %% ------------ PLOT/PRINT ALL LOCATION-BASED ROC CURVES AND AUC

plot_roc_curve(df_sydney_test_class, df_sydney_test_pred_rf, df_sydney_test_pred_lr,
               df_sydney_test_pred_knn, df_sydney_test_pred_svm)
print_auc('Sydney', df_sydney_test_class, df_sydney_test_pred_rf, df_sydney_test_pred_lr,
          df_sydney_test_pred_knn, df_sydney_test_pred_svm)

plot_roc_curve(df_perth_test_class, df_perth_test_pred_rf, df_perth_test_pred_lr,
               df_perth_test_pred_knn, df_perth_test_pred_svm)
print_auc('Perth', df_perth_test_class, df_perth_test_pred_rf, df_perth_test_pred_lr,
          df_perth_test_pred_knn, df_perth_test_pred_svm)

plot_roc_curve(df_canberra_test_class, df_canberra_test_pred_rf, df_canberra_test_pred_lr,
               df_canberra_test_pred_knn, df_canberra_test_pred_svm)
print_auc('Canberra', df_canberra_test_class, df_canberra_test_pred_rf, df_canberra_test_pred_lr,
          df_canberra_test_pred_knn, df_canberra_test_pred_svm)

plot_roc_curve(df_adelaide_test_class, df_adelaide_test_pred_rf, df_adelaide_test_pred_lr,
               df_adelaide_test_pred_knn, df_adelaide_test_pred_svm)
print_auc('Adelaide', df_adelaide_test_class, df_adelaide_test_pred_rf, df_adelaide_test_pred_lr,
          df_adelaide_test_pred_knn, df_adelaide_test_pred_svm)

plot_roc_curve(df_brisbane_test_class, df_brisbane_test_pred_rf, df_brisbane_test_pred_lr,
               df_brisbane_test_pred_knn, df_brisbane_test_pred_svm)
print_auc('Brisbane', df_brisbane_test_class, df_brisbane_test_pred_rf, df_brisbane_test_pred_lr,
          df_brisbane_test_pred_knn, df_brisbane_test_pred_svm)
