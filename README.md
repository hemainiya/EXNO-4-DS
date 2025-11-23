# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df.head()

```

<img width="358" height="247" alt="image" src="https://github.com/user-attachments/assets/70d99beb-56a2-4a3e-a479-c51fca81dc2c" />

```

df_null_sum=df.isnull().sum()
df_null_sum

```

<img width="157" height="252" alt="image" src="https://github.com/user-attachments/assets/da15e811-57f9-4f4c-89ab-aae344ed4312" />

```

df.dropna()

```

<img width="352" height="503" alt="image" src="https://github.com/user-attachments/assets/a2c8adfb-b344-4c56-9c42-56e879f95aad" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```

<img width="170" height="179" alt="image" src="https://github.com/user-attachments/assets/ef58f23f-7d79-4aba-ae47-64b58210e9b6" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```

<img width="333" height="251" alt="image" src="https://github.com/user-attachments/assets/a9a3c857-322a-4d9d-8d1e-e4319b4cdf28" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

<img width="433" height="434" alt="image" src="https://github.com/user-attachments/assets/71e42acd-0493-46f7-8ea8-fc4186065fdd" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="433" height="454" alt="image" src="https://github.com/user-attachments/assets/8960d272-10f6-4b6a-8975-9f0f2e2eaa58" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

<img width="402" height="531" alt="image" src="https://github.com/user-attachments/assets/a46521bf-5ca3-4dda-b29c-edc5a95d8992" />

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```

<img width="386" height="245" alt="image" src="https://github.com/user-attachments/assets/bf195d39-7714-45bb-b5f1-dead9819b52c" />

```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```

<img width="462" height="454" alt="image" src="https://github.com/user-attachments/assets/6ca3d8d0-cd65-4e2c-a299-f3bd54e55e08" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```

<img width="267" height="574" alt="image" src="https://github.com/user-attachments/assets/d675a1b5-5500-4244-a717-4fbba5eda6dd" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

<img width="1101" height="528" alt="image" src="https://github.com/user-attachments/assets/0d281199-27b7-47d0-aabf-66b1a29bbb0b" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="907" height="519" alt="image" src="https://github.com/user-attachments/assets/08ff63c3-7452-472d-b21b-dba42352d8fd" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

<img width="454" height="91" alt="image" src="https://github.com/user-attachments/assets/30d1ae14-c2d2-46e1-8e8e-fb117a4025b3" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```

<img width="476" height="452" alt="image" src="https://github.com/user-attachments/assets/e6c356f7-8fd6-4e03-a54d-07a0e82e3bbe" />

```
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

<img width="1202" height="530" alt="image" src="https://github.com/user-attachments/assets/0e5347d6-2983-4189-b4d1-aeeacf4ce920" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="1007" height="525" alt="image" src="https://github.com/user-attachments/assets/3c217fee-1086-475c-8b2b-1fb77b1a5a8c" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```

<img width="897" height="106" alt="image" src="https://github.com/user-attachments/assets/e3e5f5a3-dae6-48ba-903e-18c23fce98d8" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

<img width="476" height="99" alt="image" src="https://github.com/user-attachments/assets/34d330e0-dcf5-4cfd-9323-7f173e2bf360" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```

<img width="582" height="56" alt="image" src="https://github.com/user-attachments/assets/2a6607d2-2c4d-4320-8292-1fa74d546170" />

```
!pip install skfeature-chappers
```

<img width="1582" height="386" alt="image" src="https://github.com/user-attachments/assets/59c945dc-101a-40a5-951f-62d404f63413" />

```
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```

<img width="1028" height="520" alt="image" src="https://github.com/user-attachments/assets/f98ecf28-3a2f-4888-9e6d-a338fe9f3cba" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 k_anova = 5
 selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
 X_anova = selector_anova.fit_transform(X, y)
 selected_features_anova = X.columns[selector_anova.get_support()]
 print("\nSelected features using ANOVA:")
 print(selected_features_anova)
```

<img width="901" height="143" alt="image" src="https://github.com/user-attachments/assets/8e1ffb17-9e75-4a33-ba76-ae2945fba3fd" />

```
 # Wrapper Method
 import pandas as pd
 from sklearn.feature_selection import RFE
 from sklearn.linear_model import LogisticRegression
 df=pd.read_csv("/content/income(1) (1).csv")
 # List of categorical columns
 categorical_columns = [
 'JobType',
 'EdType',
 'maritalstatus',
 'occupation',
 'relationship',
 'race',
 'gender',
 'nativecountry'
 ]
 # Convert the categorical columns to category dtype
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```

<img width="1187" height="576" alt="image" src="https://github.com/user-attachments/assets/d1a147ad-5ddc-4d6a-97b9-38ddb1b7f983" />

```
X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 logreg = LogisticRegression()
 n_features_to_select =6
 rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
 rfe.fit(X, y)
```

<img width="1373" height="1049" alt="image" src="https://github.com/user-attachments/assets/5a6debf0-be6f-4370-b570-50819157e177" />

<img width="311" height="188" alt="image" src="https://github.com/user-attachments/assets/84429465-f542-4627-a2f7-acebc64eeb0f" />

# RESULT:
   The given data and perform Feature Scaling and Feature Selection process and save the data to a file is succussfully verified.

