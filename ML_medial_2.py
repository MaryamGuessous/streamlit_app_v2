#Step 1: Import the required modules
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz,DecisionTreeRegressor
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.metrics import accuracy_score
import seaborn as seabornInstance 
import statsmodels.api as sm
import pydot
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz,DecisionTreeRegressor

from sklearn.metrics import mean_squared_error as MSE

from sklearn.tree import DecisionTreeClassifier, export_graphviz,DecisionTreeRegressor
import graphviz

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import export_graphviz

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

from numpy import arange
from pandas import read_csv
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold

from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso

from numpy import arange
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso

import warnings
import pickle
warnings.filterwarnings("ignore")


inputFileName = "donnees_synthese_1706.xlsx"
data = pd.read_excel(inputFileName)

inputFileName_2 = "donnees_synthese_2_1606.xlsx"
data_2 = pd.read_excel(inputFileName_2)

data= pd.get_dummies(data, columns=['Sexe'])
data_2 = pd.get_dummies(data_2, columns=['Sexe'])

data_2= pd.get_dummies(data_2, columns=['Categorie Age'])

data= data.drop([ 'Patient', 'Reference'], axis=1)
data_2= data_2.drop([ 'Patient', 'Reference'], axis=1)

High_Fibro=data['FibroTest'].mean() + 2.3*data['FibroTest'].std()
Low_Fibro=data['FibroTest'].mean() - 2.3*data['FibroTest'].std()

High_Acti=data['Actitest'].mean() + 2*data['Actitest'].std()
Low_Acti=data['Actitest'].mean() - 2*data['Actitest'].std()

data = data[(data['FibroTest']<High_Fibro) & (data['FibroTest']>Low_Fibro)]
data = data[(data['Actitest']<High_Acti) & (data['Actitest']>Low_Acti)]


X= data.drop([ 'FibroTest', 'Actitest'], axis=1)
Y_F = data[['FibroTest']]
Y_A = data[['Actitest']]

#Let's divide the datased into training and testing sets
import random
random.seed(0)
X_train_F, X_test_F, y_train_F, y_test_F = train_test_split(X, Y_F, test_size=0.3, random_state=10)
X_train_A, X_test_A,  y_train_A, y_test_A = train_test_split(X, Y_A, test_size=0.3, random_state=43)

from sklearn.metrics import r2_score

#Regression Linéaire avec la variable Age
LinReg_F = LinearRegression()
LinReg_F.fit(X = X_train_F, y = y_train_F)

LinReg_A = LinearRegression()
LinReg_A.fit(X = X_train_A, y = y_train_A)

pickle.dump(LinReg_F,open('model_F.pkl','wb'))
pickle.dump(LinReg_A,open('model_A.pkl','wb'))
