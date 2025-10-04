import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

loan=pd.read_csv(r"C:\Users\Omar\Documents\archive\loan_approval_dataset.csv")

loan=loan.dropna()
loan['loan/income']=loan[' loan_amount']/loan[' income_annum']
loan[' self_employed'] = loan[' self_employed'].map({' No': 0, ' Yes': 1})
loan[' education'] = loan[' education'].map({' Not Graduate': 0, ' Graduate': 1})
loan[' loan_status'] = loan[' loan_status'].map({' Rejected': 0, ' Approved': 1})
loan['assets']=loan[' commercial_assets_value']*loan[' residential_assets_value']*loan[' luxury_assets_value']*loan[' bank_asset_value']

x=loan[[' self_employed',' education','assets','loan/income',]]
y=loan[' loan_status']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)
smote=SMOTE()
x_train_resample,y_train_resample=smote.fit_resample(x_train,y_train)

model=LogisticRegression(class_weight='balanced')

model.fit(x_train_resample,y_train_resample)
y_pred=model.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(f1_score(y_test,y_pred))

unique_test, counts_test = np.unique(y_test, return_counts=True)
plt.pie(counts_test, labels=[f"Actual {u}" for u in unique_test], 
        autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title("Distribution of Actual Loan Status (y_test)")
plt.show()

unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
plt.pie(counts_pred, labels=[f"Predicted {u}" for u in unique_pred], 
        autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'orange'])
plt.title("Distribution of Predicted Loan Status (y_pred)")
plt.show()