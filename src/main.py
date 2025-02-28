""" Importing Libraries """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns', None)

""" Loading & Exploring Data """

data = pd.read_csv('./datasets/Telco-Customer-Churn.csv')
# print(data.head())
# for col in data.columns:
#     print(col, " - ", data[col].unique())

""" Preparing Data """

data = data.drop(['customerID'], axis=1)
data['TotalCharges'] = data['TotalCharges'].replace({" ": '0.0'})
data['TotalCharges'] = data['TotalCharges'].astype(float)

""" Data Preprocessing """

data['Churn'] = data['Churn'].map({"Yes": 1, "No": 0})

# Label Encoding the categorical data
le = LabelEncoder()

categorical_columns = data.select_dtypes(include='object').columns
encoders = {}

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

""" Visualising Numeric Data """

def plot_subplot(col1, col2, col3):
    figure, ax = plt.subplots(3, 1)

    ax[0].hist(data[col1])
    ax[0].axvline(data[col1].mean(), color='#96e072', label='Mean')
    ax[0].axvline(data[col1].median(), color='#f67e7d', label='Median')
    ax[0].legend()

    ax[1].hist(data[col2])
    ax[1].axvline(data[col2].mean(), color='#96e072', label='Mean')
    ax[1].axvline(data[col2].median(), color='#f67e7d', label='Median')
    ax[1].legend()

    ax[2].hist(data[col3])
    ax[2].axvline(data[col3].mean(), color='#96e072', label='Mean')
    ax[2].axvline(data[col3].median(), color='#f67e7d', label='Median')
    ax[2].legend()

    plt.tight_layout()
    plt.savefig('visualizations/numeric_data_subplot.png', dpi=300)
    plt.show()

def plot_heatmap(df):
    sns.heatmap(df.corr(), annot=True, cmap='Blues', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.savefig("visualizations/numeric_data_heatmap.png")
    plt.show()

plot_subplot("tenure", "MonthlyCharges", "TotalCharges")
plot_heatmap(data[["tenure", "MonthlyCharges", "TotalCharges"]])

""" Splitting Features and Target Data """

X = data.drop(['Churn'], axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

""" Oversampling because of large data difference """

smote = SMOTE(random_state=0)
X_train, y_train = smote.fit_resample(X_train, y_train)

""" Building Model """

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=0),
    'Random Forest': RandomForestClassifier(random_state=0),
    'XGBoost': XGBClassifier(random_state=0)
}

for model_name, model in models.items():
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f'Model: {model_name} ——> Cross Validation Score: {np.mean(cv_score):.2f}')
print('—'*60)

# proceeding with RandomForestClassifier as it has the highest cross validation accuracy score
print(f'\nRandom Forest Classifier:')
model_rfc = RandomForestClassifier(random_state=0)
model_rfc.fit(X_train, y_train)

""" Evaluating Model using test data """

y_prediction = model_rfc.predict(X_test)

acc_score = accuracy_score(y_test, y_prediction)
conf_matrix = confusion_matrix(y_test, y_prediction)

print(f'Accuracy Score: {acc_score:.2f}')
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Purples', fmt='.2f')
plt.title("Random Forest Classifier\nConfusion Matrix ")
plt.savefig('visualizations/rfc_confusion_matrix.png', dpi=200)
plt.show()

""" Saving Model as Pickle file """

with open('churn_classifier_rfc_model.pkl', 'wb') as f:
    pickle.dump(model_rfc, f)