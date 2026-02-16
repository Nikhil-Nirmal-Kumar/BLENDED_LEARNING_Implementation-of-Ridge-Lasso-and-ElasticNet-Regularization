# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the car price dataset into Python using Jupyter Notebook.
2. Preprocess the data and generate polynomial features.
3. Build pipelines using Ridge, Lasso, and ElasticNet regression models.
4. Build pipelines using Ridge, Lasso, and ElasticNet regression models.

## Program:
```
#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#Load the dataset
data=pd.read_csv('encoded_car_data (1).csv')
data.head()

#Data preprocessing
#data = data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data, drop_first=True)

#Splitting the data into features and target variable
x=data.drop('price',axis=1)
y=data['price']

#Standardizing the features
scaler=StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(y.values.reshape(-1,1))

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

##Define the models and pipelines
models={
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}

#Dictionary to store results
results={}

#Train and evaluate each model
for name,model in models.items():
    #Create a pipeline with polynomial features and the model
    pipeline=Pipeline([
        ('poly',PolynomialFeatures(degree=2)),
        ('regressor',model)
    ])
    
    #Fit the model
    pipeline.fit(x_train, y_train)
    
    #Make predictions
    predictions=pipeline.predict(x_test)
    
    #Calculate performance metrics
    mse=mean_squared_error(y_test,predictions)
    mae=mean_absolute_error(y_test,predictions)
    r2=r2_score(y_test,predictions)
    
    #Store results
    results[name]={'MSE':mse,'R2 Score':r2}
    
#Print results
print('Name: Nikhil Nirmal Kumar')
print('Reg. No: 212225230201')
for model_name, metrics in results.items():
    print(f"{model_name}-Mean Squared Error: {metrics['MSE']:.2f},R2 Score:{metrics['R2 Score']:.2f}")

#Visualization of the results
#Convert results to DataFrame for easier plotting
results_df=pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index':'Model'},inplace=True)

#Set the figure size
plt.figure(figsize=(12,5))

#Bar plot for MSE
plt.subplot(1,2,1)
sns.barplot(x='Model',y='MSE',data=results_df,palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)

#Bar plot for R2 Score
plt.subplot(1,2,2)
sns.barplot(x='Model',y='R2 Score',data=results_df,palette='viridis')
plt.title('R2 Score')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)

#Show the plots
plt.tight_layout
plt.show()
```

## Output:
<img width="1276" height="749" alt="image" src="https://github.com/user-attachments/assets/3f2d82b6-f687-460d-90b6-45bbf601fd03" />


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
