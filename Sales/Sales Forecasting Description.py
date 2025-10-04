import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

train=pd.read_csv(r"C:\Users\Omar\Downloads\archive (2)\train.csv")
test=pd.read_csv(r"C:\Users\Omar\Downloads\archive (2)\test.csv")
features=pd.read_csv(r"C:\Users\Omar\Downloads\archive (2)\features.csv")
stores=pd.read_csv(r"C:\Users\Omar\Downloads\archive (2)\stores.csv")

features=features.dropna()
merged0=train.merge(stores,how="left",on="Store")
train_final=merged0.merge(features,how="left",on=["Store","Date","IsHoliday"])
train_final=train_final.sort_values(by=["Date","Store"],ascending=True)
train_final['IsHoliday']=train_final['IsHoliday'].astype(str).str.strip().str.lower()
train_final['IsHoliday']=train_final['IsHoliday'].map({'false': 0, 'true': 1})

train_final["Date"] = pd.to_datetime(train_final["Date"])
train_final["Day"]=train_final["Date"].dt.day
train_final["Month"]=train_final["Date"].dt.month
train_final["Lag_7"]=train_final.groupby("Store")["Weekly_Sales"].shift(7)
train_final["Lag_80"]=train_final.groupby("Store")["Weekly_Sales"].shift(80)
train_final["Rolling_Month"] = train_final.groupby("Store")["Weekly_Sales"].rolling(window=30).mean().reset_index(level=0, drop=True)
train_final["Rolling_80"] = train_final.groupby("Store")["Weekly_Sales"].rolling(window=80).mean().reset_index(level=0, drop=True)
train_final=train_final.dropna()

x=train_final[['Day','Month','Lag_7','Lag_80','Rolling_Month','Rolling_80','IsHoliday',
               'Size','Temperature','CPI']]
y=train_final['Weekly_Sales']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(mean_squared_error(y_test,y_pred),r2_score(y_test,y_pred))
print(x_test.shape,y_test.shape,y_pred.shape)

plt.figure(figsize=(10,5),dpi=250)
plt.plot(y_test.to_numpy(), label="Actual", color="blue")
plt.plot(y_pred, label="Predicted", color="red")
plt.legend()
plt.title("Sales Forecasting Over Time (first 200 samples)")
plt.savefig('Sales Forecasting Description.jpg',dpi=250)
plt.show()

train_final.to_csv('train_final.csv', index=False)