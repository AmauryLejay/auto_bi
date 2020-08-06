## AUTO_BI : bridging the gap between costly software and time consuming fully customized BI reports 

### Why 

Inspired by my own experience in Data Science, I created auto_bi with the aim of **combining the flexibility and intelligence of fully customized analysis with the ease-of-use and time saving approach of pre-developped data analysis softwares**

Past few years trends have shown a tremendous penetration of data driven decision making in companies. This triggered the democratization of new skills within the workforce including python programming language and various data analysis and data science technics. 
In their daily work, data scientists, engineers and analysts are faced with various datasets which differ in their structure and content but often require similar pre-processing operations and aim at the same objectives: finding basic insights with respect to a target measurement and also odd ones. 
In addition, python is becoming a mainstream language and has recently started to break throught the sillo of engineering schools to reach all companies' departments, business schools, high schools and so on. Developpers want to get to the insights quicker and quicker, and don't necessarily possess the skills and time to achieve their goals anymore.

**For all these python users, the tool aims at saving time-to-insights delivery, without spending a single dollar, while keeping the flexibility to build a fully customized solution.** 

### How to use it ? 

- Clone the project

- Then from the terminal run in the folder of the project where you stored your dataset as well. Arguments are detailled in the next point.
```python main.py ./dataset/titanic.csv Survived ['Pclass','Sex','SibSp','Embarked'] ./config.yml```

- Alternatively, run the following lines in a new file or simply run the jupyer tutorial [here](https://github.com/AmauryLejay/auto_bi/blob/master/tutorial.ipynb)
```
# Imports
from categorical_target_variable import categorical_target_variable
import pandas as pd

# Input
df = pd.read_csv("dataset/titanic.csv")
target = "Survived" 
categorical_columns = ['Pclass','Sex','SibSp','Embarked']
config_path='./config.yml'

# main run 
cat = categorical_target_variable(df,target,categorical_columns,numerical_col,config_path = config_path)
cat.analyze()
```

- Have fun! 

### Already Done

##### Analysis
- [x] Basic KPIs computation
- [x] automatic column ID identification
- [x] Handling of both numerical and categoricla features
- [x] Missing Values identification and threshold
- [x] Combination of features into polynomial features
- [x] Identification of highly correlated features exceeding threshold
- [x] Identification of most relevant features with regard to target feature

##### Timestamps
- [x] Identification ideal level of granularity for representation (seconds, minutes, hours, days etc)
- [x] Identification of top growth and decrease periods

##### Visualization
- [x] Interactive Dash report 
- [x] Basic KPIs (1/5)
- [x] Most relevant features with regard to target feature

[screenshot_1](https://github.com/AmauryLejay/auto_bi/blob/master/image/screenshot_1.png)

[screenshot_2](https://github.com/AmauryLejay/auto_bi/blob/master/image/screenshot_2.png)


### To Do and Roadmap

##### Analysis
- [ ] Missing value handling strategy to choose and implement
- [ ] Highly correlated features handling
- [ ] Handling of large Data Frame / Paralization 
- [ ] Handling cases where there isn't any target column to maximise or minimise (ex sales) such as : 
	- [ ] number of rows to maximise or minimise (all rows are visits) 
	- [ ] number of specific rows to maximise or minimise (some rows are purchases, others just visits) 

- [ ] Develop more applied analysis with regards to sales column:
	- [ ] Customer Life time value
	- [ ] Churn and retention rates
	- [ ] Others 

- [ ] Timestamps:

- [ ] Feature engineering: 
	- [ ] Text features to clean

##### Visualization
- [ ] Automatic openning of the Dashboard in webbrowser
- [ ] Basic KPIs (4/5)
- [ ] Work on prioritization of display in the dashboard
- [ ] Handling of user inputs in the dashboard
- [ ] Time stamps
- [ ] Dowload of data from charts and tables in the dashboard

##### Packaging

- [ ] Test (always)
- [ ] Implement logging
- [ ] Package library

##### Logo

- [ ] Find a new cute logo

[minion](https://github.com/AmauryLejay/auto_bi/blob/master/image/logo_minion.png)

