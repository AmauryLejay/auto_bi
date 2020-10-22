import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m
import itertools
import seaborn as sns
import random 
import warnings

# timestamp
from dateutil import parser as dateparser # doc https://dateutil.readthedocs.io/en/stable/parser.html
from dateutil.parser import parserinfo as dateparser_info

# Visualization
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import webbrowser
from pages import (
    basic_kpis_only,
    analysis,
    )

# Config
import yaml
from box import Box # uselful to access yaml files like this config.group.subgroup instead of like this config['group']['subgroup']


# TODO: _display_basic_kpi + compare pandas profilling
# TODO: improve handling of missing values / Replace missing values strat? 
# TODO: Handle case where there isn't any numerical or cat features 
# TODO: Decide to potentially remove the features that are too correlated
# TODO: Handle case where the main ID column for instance is changed, the functions that depend on its value need all to be re-run DONE 

# TODO: Numerical target value 
# TODO: Handle entropy calculation when the target is not categorical DONE
# TODO: Numerical Value correlation / growth 

# TODO: Server visualisation (Dask)
# TODO: Large Data Frame handling / Paralization
# TODO: Library deployment
# TODO: Implement logging

# TODO: Target is not a column that we try to maximise, minimise or find correlation to, but rather a number of rows
# TODO: Visit Maximisation target value

# TODO: Feature Engineering Generator
# TODO: Decide how many features we choose for text
# TODO: Feature extractor (text) DONE

class timestamp_handling(categorical_target_variable):
	"""Class gathering the functions that handles datasets with timestamps"""

	def __init__(self, dataframe, target_variable, timestamp_column):
		super().__init__(dataframe, target_variable, timestamp_column)

	def timestamp_processing(self, list_granularity = ['year','month','weekday','hour','minute']):
	    """
	    Convert the date column as a timestamp if it's not as such already. Then create a year, month, weekday, hour, minute columns
	    input: 
	        - df : pandas dataframe
	        - timestamp_column : string 
	    output:
	        - df : pandas dataframe with the new columns
	    """
	    
	    # Checking if the date column is in the right date format or if it needs to be processed
	    if str(self.df[timestamp_column].dtype) != "datetime64[ns]":
	        self.df[timestamp_column] = self.df.apply(lambda row: dateparser.parse(row[timestamp_column], parserinfo=date_parser_info, dayfirst = False),axis = 1)
	    
	    # Creating the date only column, without timestamps
	    self.df['date_only'] = self.df[timestamp_column].apply(lambda row: row.date())
	    
	    # Creating column to isolate specific timestamps
	    if "year" in list_granularity: 
	        self.df['year'] = self.df[timestamp_column].apply(lambda row: row.year)
	    if "month" in list_granularity: 
	        self.df['month'] = self.df[timestamp_column].apply(lambda row: row.month)
	    if "weekday" in list_granularity: 
	        self.df['weekday'] = self.df[timestamp_column].apply(lambda row: row.weekday())
	        self.df['weekend'] = 0
	        self.df.loc[self.df.weekday > 4, 'weekend'] = 1
	    if "hour" in list_granularity: 
	        self.df['hour'] = self.df[timestamp_column].apply(lambda row: row.hour)
	    if "minute" in list_granularity:
	        self.df['minute'] = self.df[timestamp_column].apply(lambda row: row.minute)
	    
	    return self.df

	def identify_ideal_granularity_visualisation(df, min_threshold_dic):
	    """Check for each granularity level of time (year, month, day etc) how much is present in the dataset
	    input : dataframe, minimum threshold dictionnary
	    output : dictionnary such as the one below
	        {'year': {'consecutive_nbr': 5, 'ratio': 1},
	         'month': {'consecutive_nbr': 12, 'ratio': 1.0},
	         'weekday': {'consecutive_nbr': 5, 'ratio': 0.7142857142857143}}
	         
	         If the scale element doesn't meet the required threshold set in the min_threshold_dic (i.e. hour in the example above)
	         then it won't appear in the output dictionnary """
	    
	    list_granularity_option = ['year','month','weekday','hour']# ,'seconds','other',
	    scales = {}
	    
	    def _check_cycle_threshold(df, element, scales):
	        """Build for a given scale (week, month, hour etc) the ratio of represented elements (i.e. 4 months out of 12, 5 days out of 7 etc)
	        and check if that ratio passes a given threshold
	        input: 
	            - dataframe
	            - string - element (year, month, day, hour ...)
	            - dictionnary - scales"""
	        
	        def _check_nbr_consecutive_elements(nbr_consecutive_elements, scale, scales):
	            """calculate the ratio and add it to the output dictionnary scale
	            input: nbr of consecutive month, days etc
	            output: dictionnary - scale updated with ratio information if the minimum threshold is passed 
	            """
	            if nbr_consecutive_elements >= min_threshold_dic[scale]['threshold']:
	            
	                ratio = float(nbr_consecutive_elements/min_threshold_dic[scale]['max'])
	                scales[scale] = {'consecutive_nbr':nbr_consecutive_elements, 'ratio':ratio}

	            return scales 
	        
	        list_unique_elements = sorted(df[element].unique()) #HERE
	        nbr_consecutive_elements = 0

	        for element_value in list_unique_elements:
	              # Doesn't handle the case where we have 2020-01, 2020-02, 2020-03, 2020-12, will consider that 2020-12 is part of 2019, and therefore the months consecutive
	              # TODO : change that HERE  
	            try: 
	                # Checking that the two months are consecutive
	                if element_value+1 == list_unique_elements[list_unique_elements.index(element_value)+1]: #HERE
	                    nbr_consecutive_elements +=1
	                # restarting the counter and keep searching for consecutive months
	                else:
	                    nbr_consecutive_elements = 0

	            # Excluding the last month that cause an out of range error
	            except IndexError: 
	                nbr_consecutive_elements +=1
	                
	            # Checking if at this point (not at the end of the list of unique month, days etc yet) we have enough consecutive element to validate this scale 
	            scales = _check_nbr_consecutive_elements(nbr_consecutive_elements, element, scales)
	        # if we managed to get to the end of the list of elements to check without issue, check that we have found enough to validate the scale
	        scales = _check_nbr_consecutive_elements(nbr_consecutive_elements, element, scales)
	        
	        return scales
	    
	    for scale in list_granularity_option:
	        # Check that there is a cycle (i.e. minimum of two elements - year, month, week etc)
	        
	        if len(df[scale].unique()) > 2:
	            # Check that at least threshold of the cycle is represented
	            # except for Year that doesn't require a minimum threshold (12 months make a year cycle but no minimum number of year make a cycle - assuming we don't care about decades)
	            if scale =='year':
	                scales['year'] = {'consecutive_nbr':len(df[scale].unique()), 'ratio':1}
	            
	            elif scale == 'month':
	                _check_cycle_threshold(df, scale, scales)
	                
	            elif scale == 'weekday':
	                _check_cycle_threshold(df, scale, scales)
	                
	            elif scale == 'hour':
	                _check_cycle_threshold(df, scale, scales)
	            
	            elif scale == 'minute':
	                _check_cycle_threshold(df, scale, scales)
	                            
	    return scales

	def find_largest_delta(df, column_of_interest, delta = "increase", display_top_x_values = 10, rolling = 1):
	    """ identify the periods of longest consecutive growth or decrease
	    input:
	        - dataframe - main dataframe
	        - string - column of interest that has the values we want to evaluate
	        - delta - default is increase in which case we look for period of consecutive GROWTH, set it to "decrease"
	          to look for period of consecutive decrease.
	        - int - display_top_x_values sets the number of values we want to have
	        - int - rolling sets the rolling period used to calculate the mean value.
	    output: 
	        - list of tuples with ((index_start,index_end), delta value)
	        [((528, 571), 72.03633333333357),
	         ((660, 699), 40.292333333333204),
	         ((173, 203), 31.129333333333363)] """
	    
	    start = 0
	    end = 0
	    max_start = 0 
	    max_end = 0
	    diff_end_start_values = 0
	    all_consecutive_delta = {}
	    
	    # Setting dates index 
	    df.set_index(date_column,inplace = True)
	    
	    rol_mean = df[column_of_interest].astype(float).rolling(rolling).mean() # Applying rolling mean on values to remove noise and measure longer term increase or decrease 
	    val = [value for value in list(rol_mean) if str(value) != 'nan'] # List of the values (i.e. transpose of the column of interest)
	    rol_mean.plot() 
	        
	    if delta == "decrease": # Case where we look for the longest consecutive decrease
	        for i in range(0,len(val)-1): # iterating over all the values
	            if val[i] > val[i+1]: # Next value is smaller, good, we keep going
	                end = i+1
	            else: # We are hitting a reversal point in the values, therefore we record the decrease period we just traversed
	                diff_start_end_values = val[start] - val[end]
	                all_consecutive_delta[(start,end)] = diff_start_end_values
	                start = i+1 
	                end = i+1

	        # Resetting index back to the initial form
	        df.reset_index(inplace = True)

	        return sorted(all_consecutive_delta.items(), key=lambda x: x[1], reverse=True)[:display_top_x_values]

	    else: # delta = "increase" default where we look for the longest consecutive decrease
	        for i in range(0,len(val)-1): # iterating over all the values
	            if val[i] < val[i+1]: # Next value is bigger, good, we keep going
	                end = i+1
	            else: # We are hitting a reversal point in the values, therefore we record the increase period we just traversed
	                diff_end_start_values = val[end] - val[start]
	                all_consecutive_delta[(start,end)] = diff_end_start_values
	                start = i+1 
	                end = i+1

	        # Resetting index back to the initial form
	        df.reset_index(inplace = True)

	        return sorted(all_consecutive_delta.items(), key=lambda x: x[1], reverse=True)[:display_top_x_values]
# TODO Launch the timestamps function in the categorical_target_variable if there is indeed a timestamp present


class categorical_target_variable():
	"""Main class so far, handling dataset that possess a categorical target variable to study (example famous titanic survived feature [0,1]).
	The main function to call within it is analyse"""

	def __init__(self, dataframe, target_variable, categorical_columns, numerical_col, id_col = None, timestamp_column = None, config_path='./config.yml'):
		
		# Mandatory parameters
		self.df = dataframe # DataFrame
		self.target = target_variable # string
		self.categorical_columns = categorical_columns # List
		self.numerical_col = numerical_col # List
		
		# Optional parameters
		self.config_path = config_path
		self.id_col = id_col
		self.timestamp_column = timestamp_column

		# Will be generated throughout the analysis
		self.df_head = None
		self.df_shape = None
		self.df_target_count = None
		self.df_target_describe = None
		self.list_highly_correlated_features = None # List
		self.categorical_columns_expended =  None # List 
		self.df_feature_entropy = None # DataFrame
		self.list_new_categorical_features = [] # List
		self.dic_groupby_df = {}  # Dictionnary

		# Loading YAML config files
		try:
			with open(self.config_path, 'r') as ymlfile:
				self.config = Box(yaml.safe_load(ymlfile))
		except Exception as e:
		    print('Error reading the config file')

	def basic_kpi(self):
		"""Calculate basic KPIs for our dataframe and target variable column
		
		input: None 

		output: 
		- dataframe : head of the main DataFrame
		- tuple : (# of row, # of columns) of the maint DataFrame 
		- dataframe : count of the different class of the target variable 
		- dataframe : basic stats of the target variable column"""

		self.df_head = self.df.head()
		self.df_shape = self.df.shape
		self.df_target_count = pd.DataFrame(self.df[self.target].value_counts())
		self.df_target_describe = self.df[self.target].describe()

		return self.df_head, self.df_shape, self.df_target_count, self.df_target_describe

	def missing_values(self):
	    """Defines what should be the most likely column to use as unique ID
	    	Check that threshold of missing value are not met both for the ID column and for the other columns
	
		input: None 
		   
	    output: 
	    - string : Name of the ID column to use
	    - Boolean : True if too many missing value according to the threshold set in config, False otherwise.
	    - dictionnary : missing value dict {column name : number of missing values}
	    - list : name of the columns with missing values"""
	   
	    missing_val_serie = self.df.isnull().sum().sort_values()
	    
	    # Choosing the main column to use as primary key 
	    # Added the OPTIONAL setting of id column as it is possible that the user already manually inputed an ID column
	    
	    if self.id_col == None: 
	    	self.id_col = missing_val_serie.index[0]

	    number_of_missing_value_main_col = missing_val_serie[0]
	    # Calculating missing value ratio
	    ratio = number_of_missing_value_main_col / self.df.shape[0]
	    
	    # Checking if the missing value ratio is exceeding a threshold or not
	    threhold_met = False
	    if ratio >= self.config.threshold_main_col:
	        threhold_met = True 
	    
	    col_with_missing_value = [col for col in self.df.columns.to_list() if self.df[col].isnull().sum()/self.df.shape[0] > self.config.threshold_other_col] 

	    return self.id_col,threhold_met,missing_val_serie.to_dict(),col_with_missing_value

	def bining_numerical_columns(self):
		"""Transform all the numerical columns to categorical, using the bining method, according to the q config parameter choosing the number of quartiles
		
		input: None

		output:
		- list : names of newly created columns created from the numerical columns"""

		quantile = str(self.config.nbr_quantile)

		for col in self.numerical_col:
			self.df[f'{col}_{quantile}_quantile'] = pd.qcut(self.df[col],q= self.config.nbr_quantile)
			self.list_new_categorical_features.append(f'{col}_{quantile}_quantile')

	    # Updating the list of categorical columns
		self.categorical_columns_expended = self.categorical_columns + self.list_new_categorical_features

		return self.categorical_columns_expended

	def extract_correlated_features(self):
		"""Return features names which exceed the correlation threshold

		input: None 
		
		output: 
		- list -  names of features that could be removed"""

		# Check that the ID column has been defined, otherwise find it or ask the user to input it himself.
		if self.id_col is None:
			self.id_col,threhold_met,missing_val_serie,col_with_missing_value = self.missing_values()
			warnings.warn(f"The main ID column to use has not been identified. The method missing_values has been run  identified {self.id_col} as the appropriate ID column to use. If you don't think this is the appropriate column, you can set the approrpriate ID column name when declaring the categorical_target_variable object and then call again the extract_correlated_features method") 

		# Calculating correlation of features
		df_corr = self.df.drop([self.id_col], axis=1).corr()

		# Filtering the top triangle of the correlation matrice
		upper = df_corr.where(np.triu(np.ones(df_corr.shape), k = 1).astype(np.bool))
		
		# Setting aside the features that exceed the correlation threshold
		self.list_highly_correlated_features = [column for column in upper.columns if any(upper[column].abs() > self.config.correlation_threshold)]

		return self.list_highly_correlated_features

	def calculate_entropy(self, df_groupby):
	    """ 
	    Returns the entropy for a given feauture. This serves to isolate features that provide the biggest information gain with respect to our target variable.
	    entropy calculation inspired from here https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8

	    input: 
	    - dataframe : group by the feature that we want ot evaluate the entropy on df.groupby(by = feature)
	    
	    output:
	    - float : entropy value of the splitting criteria belonging to [0-1] Goal is to reach a low entropy, with a low diversity
	     among within the target classes with respect to the feature evaluated.""" 
	    	    
	    total_entropy = 0
	    count_total = df_groupby[self.id_col].sum()
	    
	    # Case where the splitting criteria is categorical 
	    list_splited_groups = list(df_groupby.index.unique())
	    
	    # Iterating over the various groups
	    for splitting_index in list_splited_groups:
	        df_splitted = df_groupby.loc[[splitting_index]]
	        total = df_splitted[self.id_col].sum()
	        
	        # Case where the target variable is categorical
	        # Iterating over the different target classes within the group
	        entropy_group = 0
	        for target_value in df_splitted[self.target].unique():
	            count = df_splitted[df_splitted[self.target] == target_value][self.id_col].iloc[0]
	            
	            if count == 0:
	                entropy_group -= 0
	            else :
	                entropy_group -= (count/total)*m.log(count/total,self.config.base)
	                
	        total_entropy += entropy_group*(total/count_total)
	    
	    return total_entropy

	def categorical_feature_combinations(self):
		"""Calculates the entropy with respect to the target class of each features and combination of features

		input : None 
		
		output:
		- dictionnary : keys are feature name, values are the entropy with respect to the target variable"""

		dic_features_entropy = {}

	    # Check that the ID column has been defined, otherwise find it or ask the user to input it himself.
		if self.id_col is None:
			self.id_col,threhold_met,missing_val_serie,col_with_missing_value = self.missing_values() 
			warnings.warn(f"The main ID column to use has not been identified. The method missing_values has been run and identified {self.id_col} as the appropriate ID column to use. If you don't think this is the appropriate column, you can set the approrpriate ID column name when declaring the categorical_target_variable object and then call again the extract_correlated_features method") 	    

	    # Check that the numerical features have been turned into coategoriclal features
		if self.categorical_columns_expended is None:
			self.categorical_columns_expended = self.bining_numerical_columns()
			warnings.warn(f"Not all numerical features have been converted to categorical format. The bining_numerical_columns attribute has been run and generated {str(len(self.list_new_categorical_features))} new features located in the variable self.list_new_categorical_features") 
	    
	    # To be handled
		if self.list_highly_correlated_features is None : 
				self.list_highly_correlated_features = self.extract_correlated_features() # TO BE HANDLED 

	    # Iterating over all base categorical features
		for feature in self.categorical_columns_expended:
			df_groupby = pd.DataFrame(self.df.groupby(by = [feature,self.target])[self.id_col].count())
			df_groupby.sort_values(by = self.id_col, inplace = True)
			df_groupby.reset_index(level = -1, inplace = True) 
			dic_features_entropy[feature] = self.calculate_entropy(df_groupby)
			#df_groupby['visu_axis'] = "_".join(list(df_groupby.index))#df_groupby.drop([self.id_col,self.target], axis=1).astype(str).apply('_'.join, axis=1)
			df_groupby.rename({self.id_col:"count"},axis =1,inplace = True)
			self.dic_groupby_df[feature] = df_groupby
	    
	    # Iterating over all combination of categorical features
		for combination_of_feature in itertools.combinations(self.categorical_columns_expended, 2):
			df_groupby = pd.DataFrame(self.df.groupby(by = [i for i in combination_of_feature + (self.target,)])[self.id_col].count())#.rename({self.id_col:"count"},axis =1)
			df_groupby.sort_values(by = self.id_col, inplace = True)
			df_groupby.reset_index(level = -1, inplace = True)
			dic_features_entropy[" . ".join([i for i in combination_of_feature])] = self.calculate_entropy(df_groupby)
			#df_groupby['visu_axis'] = "_".join(list(df_groupby.index))#df_groupby.drop([self.id_col,self.target], axis=1).astype(str).apply('_'.join, axis=1)
			df_groupby.rename({self.id_col:"count"},axis =1,inplace = True)
			self.dic_groupby_df[" . ".join([i for i in combination_of_feature])] = df_groupby
		
		self.df_feature_entropy = pd.DataFrame(data = dic_features_entropy.values(),index = dic_features_entropy.keys(),columns = ['entropy']).sort_values(by = 'entropy')

		return self.df_feature_entropy


	def analyse(self, visualise = True):
		"""
		Method calling all the function above, returning at the moment an ordered dictionnary with the entropy measure value for each features.
		Possible to change the ouput to other features.

		optional input : Boolean True, will launch the Dash dashboard on web server False, will only return the df_feature_entropy

		output: 
		dataframe: df_feature_entropy with column name and entropy value  
		dash interactive dashboard
		"""
		
		def _run_everything():

			# Run basic KPIS
			self.basic_kpi()

			# Then Check that the necessary functions have been called, otherwise, call them
			if self.id_col is None : 
				self.id_col,threhold_met,missing_val_serie,col_with_missing_value = self.missing_values() # self.

			if self.categorical_columns_expended is None : 
				self.categorical_columns_expended = self.bining_numerical_columns()

			if self.list_highly_correlated_features is None : 
				self.list_highly_correlated_features = self.extract_correlated_features() # TO BE HANDLED 

			if self.df_feature_entropy is None: 
				self.df_feature_entropy = self.categorical_feature_combinations()

		if visualise : 

			_run_everything()

			# Visualise
			visualisation.launch_dash_server(self)

		else : 
			_run_everything()
		
		return self.df_feature_entropy


class visualisation(categorical_target_variable):
	"""Secondary class handling all the Dash interactive dashboard settings"""

	def __init__(self, dataframe, target_variable, categorical_columns, numerical_col):
		super().__init__(dataframe, target_variable, categorical_columns, numerical_col)
		
	def launch_dash_server(self):

		app = dash.Dash(
 	    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

		server = app.server

		# Describe the layout/ UI of the app
		app.layout = html.Div(
    	[dcc.Location(id="url", refresh=False), html.Div(id="page-content")])

		# Update page
		@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
		def display_page(pathname):
			if pathname == "/auto_bi/basic_kpis_only":
				return basic_kpis_only.create_layout(app, self.df_head,self.df_target_count)
			elif pathname == "/auto_bi/analysis":
				return analysis.create_layout(app,self.dic_groupby_df, self.df_feature_entropy, self.config.entropy_threshold,self.target,headers = True)
			elif pathname == "/auto_bi/full-view":
				return (
                    basic_kpis_only.create_layout(app, self.df_head,self.df_target_count),
                    analysis.create_layout(app,self.dic_groupby_df, self.df_feature_entropy, self.config.entropy_threshold,self.target,headers = False),
                )
			else:
				return basic_kpis_only.create_layout(app, self.df_head,self.df_target_count)
		
		# Run Server
		webbrowser.open('http://127.0.0.1:8050/auto_bi/analysis')
		app.run_server(debug=True,port=8050)

	def _display_correlation(self):
		"""Display correlation matrix but not used at the moment"""

		# Creating a mask to get rid of half of the correlation matrix 
		mask = np.triu(np.ones_like(df_corr, dtype=np.bool))

		# Set size of the figure
		sns.set(rc={'figure.figsize':(7,5)})

		# Generate a visualization of the correlation matrix
		sns.heatmap(df_corr, annot=True, mask = mask,square=True, linewidths=.5)
		plt.title("Feature Correlation")
		plt.show()

		return 0
