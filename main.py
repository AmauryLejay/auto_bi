import pandas as pd
from auto_bi import categorical_target_variable

df = pd.read_csv("dataset/titanic.csv")
target = "Survived" 
categorical_columns = ['Pclass','Sex','SibSp','Embarked']
numerical_col = ['Age','Fare']

cat = categorical_target_variable(df,target,categorical_columns,numerical_col)

df_entropy = cat.analyse()

print(df_entropy.head())
