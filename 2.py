import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
df=pd.read_csv('cleaned_thyroid_data.csv')
edges=[('diagnosis',col) for col in df.columns[:-1]]
print(df.isnull().sum())
df.replace('?', np.nan, inplace=True)
print(edges)
print(df.isnull().sum())
print(df)
X = df.iloc[:, :-1]  
y = df.iloc[:, -1]   

categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns

categorical_imputer = SimpleImputer(strategy='most_frequent')  # Mode imputation for categorical
numerical_imputer = SimpleImputer(strategy='mean')            # Mean imputation for numerical


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_imputer, categorical_columns),
        ('num', numerical_imputer, numerical_columns)
    ]
)


X_imputed = preprocessor.fit_transform(X)

X_imputed_df = pd.DataFrame(X_imputed, columns=categorical_columns.append(numerical_columns))

# Display the DataFrame after imputation
print("\nDataFrame after imputation:")
print(X_imputed_df.head())

df_imputed = pd.concat([X_imputed_df, y.reset_index(drop=True)], axis=1)


df_imputed.to_csv('imputed_thyroid_data.csv', index=False)

print("\nImputed DataFrame saved as 'imputed_thyroid_data.csv'")
