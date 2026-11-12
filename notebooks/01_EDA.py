import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/nitinmishra/Documents/customer-churn-prediction/data/processed/churn_data.csv')

#1. Phase-1 :- Basic data understanding

print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.info())
print(df.dtypes)
print(df.nunique())
# Identify numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print("Numerical Columns:")
print(list(numerical_cols))

print("\nCategorical Columns:")
print(list(categorical_cols))

#2. Phase-2 :- Data Quality Checking

print(df.isnull().sum()) #Missing Values
print(df.duplicated().sum()) #duplicate records
print(df.dtypes) #incorrect data types

#3. Phase-3 :- Univariate Analysis

numerical_data = df[['SeniorCitizen','tenure', 'MonthlyCharges', 'TotalCharges']]

for col in numerical_data:
    sns.histplot(data=df, x=col, bins = 20, kde=True)

    plt.title(f'Distribution of {col}')
    print(f'Distribution of {col}')

    plt.show()

numerical_data = df[['tenure', 'MonthlyCharges', 'TotalCharges']]

for col in numerical_data:
    plt.figure(figsize = (8,5))
    sns.boxplot(data = df, x = col)
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

categorical_data = ['gender', 'Contract', 'PaymentMethod', 'Churn']
for col in categorical_data:
    sns.countplot(data = df, x = col)
    plt.title(f'countplot of {col}')
    plt.xlabel(col)
    plt.show()

#4. Bivariate Analysis

numerical_data = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numerical_data:
    plt.figure(figsize = (8,5))
    sns.boxplot(data = df, x = 'Churn', y = col)
    plt.show()

categorical_data = ['Contract','PaymentMethod','InternetService','TechSupport']
for col in categorical_data:
    plt.figure(figsize = (8,5))
    sns.boxplot(data = df, x = 'Churn', y = col)
    plt.show()

#5. Multivariate analysis

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
numerical_data = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
plt.figure(figsize = (8,5))
for col in numerical_data:
    sns.heatmap(numerical_data.corr(), annot = True, cmap = 'coolwarm', fmt = '.2f')
    plt.title('Correlation Heatmap')
    plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Churn')
plt.title('Tenure vs MonthlyCharges by Churn')
plt.xlabel('Tenure')
plt.ylabel('Monthly Charges')
plt.show()

churn_pivot = df.pivot_table(
    index='Contract',
    columns='InternetService',
    values='Churn',
    aggfunc=lambda x: (x == 'Yes').mean() * 100
)

plt.figure(figsize=(9, 5))
sns.heatmap(churn_pivot, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Churn Rate by Contract and Internet Service')
plt.ylabel('Contract')
plt.xlabel('Internet Service')
plt.show()