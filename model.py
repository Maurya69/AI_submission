import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('imputed_thyroid_data.csv')


X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

edges = [('diagnosis', col) for col in X_train.columns]


model = BayesianNetwork(edges)
model.fit(pd.concat([X_train, y_train], axis=1), estimator=MaximumLikelihoodEstimator)


inference = VariableElimination(model)
X_test=pd.DataFrame(X_test)
print(X_test)
y_pred=[]

for i in range(X_test.shape[0]):
    query_result = inference.map_query(variables=['diagnosis'], evidence=X_test.iloc[0].to_dict())
    # print('Predicted value for D:', query_result['diagnosis'])
    y_pred.append(query_result['diagnosis'])
    
y_test=y_test.values
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')