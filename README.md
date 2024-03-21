# Employee-Turnover
This project help detect employees that are likely to voluntary turnover. The model build algorithms based on the correlation patterns of data attributes of employee who left voluntarily and extends to existing employees to predict turnover

## importing all neccessary libraries/ modules for data manipulation, visualization, and Modeling

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
%matplotlib inline
import statsmodels.api as sm
import scipy.stats as stats
import seaborn as sns
from seaborn import scatterplot
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from IPython.display import Image
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
```
```python
# reading the data
df1=pd.read_csv('/home/jovyan/census_vp.csv', encoding='cp1252')
```
```python
df2=pd.read_csv('/home/jovyan/termination.csv', encoding='cp1252')
```
```python
df = pd.merge(df1, df2, on='employee_id', how='left')
```
```python
# Checking the data shape
df1.shape
```
```python
df2.shape
```
```python
df.shape
```
```python
df
```

## Data Quality Check
***

```python
df.isnull().any()
```
```python
# Checking the format of the columns
df.info()
```
```python
# Checking all columns name
df.columns
```

## Data Scrubbing
***

```python
# Renaming certain columns for better readability
df = df.rename(columns={'length_of_service':'service', 'years_in_current_position':'current','years_in_previous_role':'previous',
                        'performance_rating':'performance','talent_matrix':'talent','position_time_type':'position_type',
                        'compensation_grade':'compensation','percentage_base-change':'change','promotion_within_two_years':'promotion',
                        'marital_status':'relationship','termination_reason':'reason','termination_category':'turnover'})
```
```python
# Checking the columns name again
df.columns
```
```python
df.head()
```
```python
# Dividing the columns into discrete and continous features
numerical_feature = [feature for feature in df.columns if df[feature].dtypes !="O"]
discrete_feature = [feature for feature in numerical_feature if len(df[feature].unique()) <10]
continuous_feature = [feature for feature in numerical_feature if feature not in discrete_feature]
```
```python
continuous_feature
```
```python
# Dropping 'employee_id'
df.drop(['employee_id'], axis=1, inplace=True)
```
```python
# Move the response variable termination to the front of the table
front = df['turnover']
df.drop(labels=['turnover'], axis=1, inplace = True)
df.insert(0,'turnover', front)
```
```python
df.head(2)
```

## Data Exploration and Visualization
***

```python
# Calculating Termination Acrosss 'Generation'
gen_agg = df.groupby(['generation', 'turnover'])['position_type'].count().unstack().fillna(0)
gen_agg
```
```python
# Very simple one-linear using our agg_tips dataframe
gen_agg.plot(kind='bar', stacked = True, figsize = (10,5))
```
```python
# Adding the title and rotating the x-axis labels to be horizontal
plt.title('Termination Across Generation')
plt.xticks(rotation=90, ha='center')
```
### Distribution Plots
***

```python
# Distribution Plots (length_of_service - age - years_in_current_position)
#Set up the metplotlib figure
f, axes = plt.subplots(ncols=3, figsize=(15,6))
```
```python
# Graph Employee length_of_service
sns.distplot(df.service, kde=False, color="r", ax=axes[0]).set_title('Employee Length Of Service Distribution')
axes[0].set_ylabel('Employee Count')
```
```python
# Graph Employee years_in_current_position
sns.distplot(df.current, kde=False, color="b", ax=axes[1]).set_title('Employee Years In Current Position Distribution')
axes[1].set_ylabel('Employee Count')
```
```python
# Graph Employee years_in_previous_role
sns.distplot(df.previous, kde=False, color="g", ax=axes[2]).set_title('Employee Years In Previous Role Distribution')
axes[2].set_ylabel('Employee Count')
```
```python
#Set up the metplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15,6))
```
```python
# Graph Employee Age
sns.distplot(df.age, kde=False, color="r", ax=axes[0]).set_title('Employee Age Distribution')
axes[0].set_ylabel('Employee Count')
```
```python
# Graph Employee Percentage Base Change
sns.distplot(df.change, kde=False, color="y", ax=axes[1]).set_title('Employee Percentage Base Change Distribution')
axes[1].set_ylabel('Employee Count')
```
```python
sns.kdeplot(
   data=df, x="age", hue="generation",
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)
```
### Boxplots
```python
# Lenght of Service vs Percentage Base Change [Boxplot]
sns.boxplot(x='service', y='relationship', hue='turnover', data=df)
```
```python
# Years In Current Position vs Percentage Base Change [Boxplot]
sns.boxplot(x='current', y='position_type', hue='turnover', data=df)
```
```python
# Years In Previous Role vs Percentage Base Change [Boxplot]
sns.boxplot(x='service', y='gender', hue='turnover', data=df)
```

### Correlation Matrix and Heatmap
***

```python
# Creating the Correlation Plot among all variables
df.corr()
```
```python
# Creating the Correlation Plot among all variables
corrmat = df.corr(method = 'spearman')
plt.figure(figsize=(10,10))
#plot Heat Map
g=sns.heatmap(corrmat,annot=True)
```

### K-Means Clustering of Turnover
***
```python
# Preparing data for plotting
#Load Data
data = load_digits().data
pca = PCA(2)
```
```python
#Transform the data
df = pca.fit_transform(data)
```
```python
df.shape
```
```python
# Aplying K-Means to the data
#Initialize the class object
kmeans = KMeans(n_clusters= 10)
```
```python
#predict the labels of clusters.
label = kmeans.fit_predict(df)
```
```python
print(label)
```
```python
# Plotting Label 0 K-Means Clusters
import matplotlib.pyplot as plt
```
```python
#filter rows of original data
filtered_label0 = df[label == 0]
```
```python
#plotting the results
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
plt.show()
```
```python
# Plotting Additional K-means Clusters

#filter rows of original data
filtered_label2 = df[label == 2]
 
filtered_label8 = df[label == 8]
```
```python
#Plotting the results
plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
plt.scatter(filtered_label8[:,0] , filtered_label8[:,1] , color = 'black')
plt.show()
```
```python
# Plot All K-Means Clusters

#Getting unique labels
u_labels = np.unique(label)
```
```python
#plotting the results:
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()
```
```python
#find the centroid
centers = np.array(kmeans.cluster_centers_)
centers
centroid = pd.DataFrame(centers)
centroid
```
```python
#last we will visualizing the clustering result using seaborn based on sepalwidth and sepalLegnth

sns.scatterplot(x = True, y = True, s = 50, c = 'change', marker = 'o', hue = True)
sns.scatterplot(x = centers[:,0], y = centers[:,1], marker="o", color='r', s = 70, label="centroid")
```

## Data Pre-processing
***
### Outliers Treatment
***
```python
# Handling Outliers in 'length_of_service' columns using Inter Quantile Range (IQR) Method
IQR = df.service.quantile(0.75) - df.service.quantile(0.25)
lower_bridge = df.service.quantile(0.25) - (IQR*1.5)
upper_bridge = df.service.quantile(0.75) + (IQR*1.5)
print (lower_bridge, upper_bridge)
```
```python
# Replacing Outliers in 'length_of_service' column using Inter Quantile Range (IQR) Method
df.loc[df['service']>=23.76, 'service'] = 23.76
df.loc[df['service']<=-10.89, 'service'] = -10.89
```
```python
# Handling Outliers in 'years_in_current_position' columns using Inter Quantile Range (IQR) Method
IQR = df.current.quantile(0.75) - df.current.quantile(0.25)
lower_bridge = df.current.quantile(0.25) - (IQR*1.5)
upper_bridge = df.current.quantile(0.75) + (IQR*1.5)
print (lower_bridge, upper_bridge)
```
```python
# Replacing Outliers in 'years_in_current_position' column using Inter Quantile Range (IQR) Method
df.loc[df['current']>=10.135, 'current'] = 10.135
df.loc[df['current']<=-4.225, 'current'] = -4.225
```
```python
# Handling Outliers in 'years_in_previous_role' columns using Inter Quantile Range (IQR) Method
IQR = df.previous.quantile(0.75) - df.previous.quantile(0.25)
lower_bridge = df.previous.quantile(0.25) - (IQR*1.5)
upper_bridge = df.previous.quantile(0.75) + (IQR*1.5)
print (lower_bridge, upper_bridge)
```
```python
# Replacing Outliers in 'years_in_previous_role' column using Inter Quantile Range (IQR) Method
df.loc[df['previous']>=4.29, 'previous'] = 4.29
df.loc[df['previous']<=-2.57, 'previous'] = -2.57
```
```python
# Handling Outliers in 'age' columns using Inter Quantile Range (IQR) Method
IQR = df.age.quantile(0.75) - df.age.quantile(0.25)
lower_bridge = df.age.quantile(0.25) - (IQR*1.5)
upper_bridge = df.age.quantile(0.75) + (IQR*1.5)
print (lower_bridge, upper_bridge)
```
```python
# Replacing Outliers in 'age' column using Inter Quantile Range (IQR) Method
df.loc[df['age']>=77.0, 'age'] = 77.0
df.loc[df['age']<=29.0, 'age'] = 29.0
```
```python
# Handling Outliers in 'change' columns using Inter Quantile Range (IQR) Method
IQR = df.change.quantile(0.75) - df.change.quantile(0.25)
lower_bridge = df.change.quantile(0.25) - (IQR*1.5)
upper_bridge = df.change.quantile(0.75) + (IQR*1.5)
print (lower_bridge, upper_bridge)
```
```python
# Replacing Outliers in 'age' column using Inter Quantile Range (IQR) Method
df.loc[df['change']>= 0.075, 'change'] = 0.075
df.loc[df['change']<= -0.045, 'change'] = -0.045
```

## Missing Data Treatment and Encoding
***

```python
df.head()
```
```python
# Checking missing data
df.isnull().sum()
```
```python
# Observing the percentage of missing values in each column
df.isnull().sum()/df.shape[0]
```
```python
# Encoding the 'Termination' into 1 (Voluntary Terminated Employee) and 0 (Others)
df['turnover'] = df['turnover'].apply(lambda x:1 if x=='Terminate Employee > Voluntary' else 0)
```
```python
# Encoding label in 'promotion within two years' into 1 (Yes) and 0 (null)
df['promotion'] = df['promotion'].apply(lambda x:1 if x=='Yes' else 0)
```
```python
# Encoding labels in 'retirement_risk' into 1 (Yes) and 0 (null)
df['retirement_risk'] = df['retirement_risk'].apply(lambda x:1 if x=='Yes' else 0)
```
```python
df.shape
```
```python
# Replacing Missing Values by Mode of the Column (These are in categorical variables)***
df = df.fillna(df.mode().iloc[0])
```
```python
# Checking missing data
df.isnull().sum()
```
```python
# Encoding labels in talent column into 0 to 8
df['talent']= df['talent'].apply(lambda x: ['Box 9', 'Box 8', 'Box 7', 'Box 6', 'Box 5', 'Box 4', 'Box 3', 'Box 2', 'Box 1'].index(x))
```
```python
# Encode Labels in Column Performance Rating into 0 to 5
df['performance']= df['performance'].apply(lambda x: ['Meets', 'Exceeds'].index(x))
```
```python
# Encoding labels in Retention column into 0 to 3
df['retention']= df['retention'].apply(lambda x: ['High Risk of Loss', 'Medium Risk of Loss', 'Low Risk of Loss'].index(x))
```
```python
# Encode Labels in Column: Loss Impact
df['loss_impact']= df['loss_impact'].apply(lambda x: ['8', '7', '6', 'Low Impact to Business', '5', 'Medium Impact to Business', '2', '1', 'High Impact to Business'].index(x))
```
```python
# Encode Labels in Column Performance Rating into 0 to 5
df['successor']= df['successor'].apply(lambda x: ['No', 'Yes'].index(x))
```
```python
# Encode Labels in Column: Potential
df['potential']= df['potential'].apply(lambda x: ['Needs Review','Well-Placed','Low','Promotable: An employee with the potential to be promoted.','Medium','High Potential - Ready Now','High'].index(x))
```
```python
# Dropping 'retirement_risk', 'compensation', and 'reason'
df.drop(['compensation', 'talent', 'performance', 'reason'], axis=1, inplace=True)
```
### Dropping Columns
***
```python
# Dropping 'race', 'generation', 'relationship', and 'position_type'
df.drop(['gender','race','generation','relationship','position_type'], axis=1, inplace=True)
```
```python
# Printing columns names
df.columns
```

## Modeling
***

```python
# Overview Summary (Voluntary Termination(1) vs. Others(0))
termination_Summary = df.groupby('turnover')
termination_Summary.mean()
```
```python
# Checking if Target variable class is balanced
df['turnover'].value_counts()
```
```python
# Creating x and y variable for modeling
x = df.drop('turnover', axis=1)
y = df['turnover']
```
```python
# Checking dimensions of 'x' and 'y'
x.shape, y.shape
```

### Data Balancing
***

```python
# Since our data is imbalanced (number of examples in each class is unequally distributed),
# SMOTE will be used to change the composition of samples in the training dataset by oversampling the minority class (1).
x.head()
```
```python
# Making the Data Balanced using SMOTE Method
oversample = SMOTE()
x, y = oversample.fit_resample(x,y)
```
```python
# Using the Python counter to hold the count of each of the elements present in the container
counter = Counter(y)
print(counter)
```
```python
# Checking dimensions of 'x'
x.shape
```
```python
# Applying PCA for dimensionality reduction
pca = PCA(0.98)
X_pca = pca.fit_transform(x)
X_pca.shape
```
```python
# Creating the Correlation Plot among all variables
df.corr()
```

### Splitting Data into Training and Testing Set
***

```python
# Splitting Data into Train and Test Set with a ratio of 80:20 for Modelling
x_train, x_test, y_train, y_test = train_test_split(X_pca,y,test_size=0.2, random_state=10)
```
### Logistic Regression Model
***

```python
# Initiating the Logistic Regression Model
lr_model = LogisticRegression()
```
```python
# Fitting/Training the Logistic Regression Model
lr_model.fit(x_train, y_train)
```
```python
# Predicting for Test Dataset using the Logistic Regression Model
y_pred_lr = lr_model.predict(x_test)
```
```python
# Predicting for Test Dataset using the Logistic Regression Model
y_pred_lr = lr_model.predict(x_test)
```
```python
# Checking the Accuracy, Precision, Recall and F1-score of the Logistic Regression Model on Test Dataset
print('Accuracy of Logistic Regression Model on Test Set: {:.2f}'.format(accuracy_score(y_test, y_pred_lr)))
print('Precision of Logistic Regression Model on Test Set: {:.2f}'.format(precision_score(y_test, y_pred_lr)))
print('Recall of Logistic Regression Model on Test Set: {:.2f}'.format(recall_score(y_test, y_pred_lr)))
print('F1-score of Logistic Regression Model on Test Set: {:.2f}'.format(f1_score(y_test, y_pred_lr)))
```
```python
accuracy_score(y_test, y_pred_lr)
```
```python
y_pred_lr
```

### Decision Trees Model
***

```python
# Initiating the Decision Tree Model
dt_model = tree.DecisionTreeClassifier()
```
```python
# Fitting/Training the Decision Tree Model
dt_model.fit(x_train, y_train)
```
```python
# Predicting for Test Dataset using the Decision Tree Model
y_pred_dt = dt_model.predict(x_test)
```
```python
# Predicting for Test Dataset using the Decision Tree Model
y_pred_dt = dt_model.predict(x_test)
```
```python
# Checking the Accuracy, Precision, Recall and F1-score of the Model on Test Dataset
print('Accuracy of Decision Tree Model on Test Set: {:.2f}'.format(accuracy_score(y_test, y_pred_dt)))
print('Precision of Decision Tree Model on Test Set: {:.2f}'.format(precision_score(y_test, y_pred_dt)))
print('Recall of Decision Tree Model on Test Set: {:.2f}'.format(recall_score(y_test, y_pred_dt)))
print('F1-score of Decision Tree Model on Test Set: {:.2f}'.format(f1_score(y_test, y_pred_dt)))
```
```python
# Visualizing the Decision Tree for Understanding
fig = plt.figure(figsize=(25,20))
_= tree.plot_tree(dt_model,
                 feature_names=x.columns,
                 class_names='turnover',
                 filled=True)
```
```python
# Prunning the Decision Tree Model
max_depth = []
acc = []
for i in range(1,30):
 dt_model = tree.DecisionTreeClassifier(max_depth=i, random_state = 42)
 dt_model.fit(x_train, y_train)
 pred = dt_model.predict(x_test)
 acc.append(accuracy_score(y_test, pred))
 max_depth.append(i)
```
```python
print(max(acc))
```
```python
depth = acc.index(max(acc)) + 1
depth
```
```python
dt_model = tree.DecisionTreeClassifier(max_depth=3, random_state = 42)
dt_model.fit(x_train, y_train)
pred = dt_model.predict(x_test)  # predictions on testing data
accuracy_score(y_test, pred)
```
```python
# Visualizing the Prunned Decision Tree
fig = plt.figure(figsize=(25,20))
_= tree.plot_tree(dt_model,
                 feature_names=x.columns,
                 class_names='turnover',
                 filled=True)
```
```python
# Viewing 'x'
df.head()
```
```python
# The model predict what employee will voluntarily terminate 
dt_model.predict([[17.88,6.01,0.00,63.0,1]])
```

### Random Forest Model
***

```python
# Initiating the Random Forest Model
rf_model = RandomForestClassifier()
```
```python
# Fitting/Training the Random Forest Model
rf_model.fit(x_train, y_train)
```
```python
# Predicting for Test Dataset using the Random Forest Model
y_pred_rf = rf_model.predict(x_test)
```
```python
# Checking the Accuracy, Prwecision, Recall and F1-score of the Model on Test Dataset
print('Accuracy of Random Forest Model on Test Set: {:.2f}'.format(accuracy_score(y_test, y_pred_rf)))
print('Precision of Random Forest Model on Test Set: {:.2f}'.format(precision_score(y_test, y_pred_rf)))
print('Recall of Random Forest Model on Test Set: {:.2f}'.format(recall_score(y_test, y_pred_rf)))
print('F1-score of Random Forest Model on Test Set: {:.2f}'.format(f1_score(y_test, y_pred_rf)))
```
```python
# Visualizing Random Forest Model for Understanding
fn=features = list(df.columns[1:])
cn=[str(x) for x in cn]

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf_model.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = False);
fig.savefig('turnover')
```
```python
# Viewing 'x'
df.head()
```
```python
# The model predict what employee will voluntarily terminate 
rf_model.predict([[4.68,4.680,0.00,40.0,1]])
```

### Naive Bayes Model
***

```python
# Initiating the Naive Bayes Model
nb_model = GaussianNB()
```
```python
# Fitting/Training the Naive Bayes Model
nb_model.fit(x_train, y_train)
```
```python
# Predicting for Test Dataset using the Naive Bayes Model
y_pred_nb = nb_model.predict(x_test)
```
```python
# Checking the Accuracy, Precision, Recall and F1-score of the Naive Bayes Model on Test Dataset
print('Accuracy of Naive Bayes Model on Test Set: {:.2f}'.format(accuracy_score(y_test, y_pred_nb)))
print('Precision of Naive Bayes Model on Test Set: {:.2f}'.format(precision_score(y_test, y_pred_nb)))
print('Recall of Naive Bayes Model on Test Set: {:.2f}'.format(recall_score(y_test, y_pred_nb)))
print('F1-score of Naive Bayes Model on Test Set: {:.2f}'.format(f1_score(y_test, y_pred_nb)))
```
```python
# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
```
```python
# The confusion matrix shows theere are 345+1821= incorrect predictions, and 770+2188=  correct predictions.
cm = confusion_matrix(y_test, y_pred_nb)
cm
```
```python
# The model predict what employee will voluntarily terminate 
nb_model.predict([[17.88,6.01,0.00,63.0,1]])
```

### Support Vector Machine Model
***

```python
# Initiating the Support Vector Machine Model
svm_model = SVC(kernel='linear')
```
```python
# Fitting/Training the Support Vector Machine Model
svm_model.fit(x_train, y_train)
```
```python
# Predicting for Test Dataset using the Support Vector Machine Model
y_pred_svm = svm_model.predict(x_test)
```
```python
# Checking the Accuracy, Precision, Recall and F1-score of the Support Vector Machine Model on Test Dataset
print('Accuracy of Support Vector Machine Model on Test Set: {:.2f}'.format(accuracy_score(y_test, y_pred_svm)))
print('Precision of Support Vector Machine Model on Test Set: {:.2f}'.format(precision_score(y_test, y_pred_svm)))
print('Recall of Support Vector Machine Model on Test Set: {:.2f}'.format(recall_score(y_test, y_pred_svm)))
print('F1-score of Support Vector Machine Model on Test Set: {:.2f}'.format(f1_score(y_test, y_pred_svm)))
```
```python
# Making predictions with our data
predictions = svm_model.predict(x_test)
print(predictions[:5])
```
```python
# Visualizing the linear function for our SVM classifier
w = svm_model.coef_[0]
b = svm_model.intercept_[0]
x_visual = np.linspace(32,57)
y_visual = -(w[0] / w[1]) * x_visual - b / w[1]

scatterplot(data = x_train, x=0, y=1, hue=y_train)
plt.plot(x_visual, y_visual)
plt.show()
```
```python
# The model predict what employee will voluntarily terminate 
svm_model.predict([[17.88,6.01,0.00,63.0,1]])
```

### Gradient Boosting Model
***

```python
# Initiating the Gradient Boosting Model
gb_model = GradientBoostingClassifier()
```
```python
# Fitting/Training the Gradient Boosting Model
gb_model.fit(x_train, y_train)
```
```python
# Predicting for Test Dataset using the Gradient Boosting Model
y_pred_gb = gb_model.predict(x_test)
```
```python
# Checking the Accuracy, Precision, Recall and F1-score of the Gradient Boosting Model on Test Dataset
print('Accuracy of Gradient Boosting Model on Test Set: {:.2f}'.format(accuracy_score(y_test, y_pred_gb)))
print('Precision of Gradient Boosting Model on Test Set: {:.2f}'.format(precision_score(y_test, y_pred_gb)))
print('Recall of Gradient Boosting Model on Test Set: {:.2f}'.format(recall_score(y_test, y_pred_gb)))
print('F1-score of Gradient Boosting Model on Test Set: {:.2f}'.format(f1_score(y_test, y_pred_gb)))
```
```python
gb_model.fit(x_train,y_train)
print(classification_report(y_test,gb_model.predictt(x_test)))
```
```python
# Get the tree number 42
sub_tree_42 = gb_model.estimators_[42, 0]
```
```python
# The model predict what employee will voluntarily terminate 
gb_model.predict([[17.88,6.01,0.00,63.0,1]])
```

### Extreme Gradient Boosting Model
***

```python
# Initiating the X-Gradient Boosting Model
xgb_model = XGBClassifier()
```
```python
# Fitting/Training the Extreme Gradient Boosting Model
xgb_model.fit(x_train, y_train)
```
```python
# Predicting for Test Dataset using XGBClassifier Model
y_pred_xgb = xgb_model.predict(x_test)
```
```python
# Checking the Accuracy, Precision, Recall and F1-score of the X-Gradient Boosting Model on Test Dataset
print('Accuracy of X-Gradient Boosting Model on Test Set: {:.2f}'.format(accuracy_score(y_test, y_pred_xgb)))
print('Precision of X-Gradient Boosting Model on Test Set: {:.2f}'.format(precision_score(y_test, y_pred_xgb)))
print('Recall of X-Gradient Boosting Model on Test Set: {:.2f}'.format(recall_score(y_test, y_pred_xgb)))
print('F1-score of X-Gradient Boosting Model on Test Set: {:.2f}'.format(f1_score(y_test, y_pred_xgb)))
```
```python
# The model predict what employee will voluntarily terminate 
xgb_model.predict([[17.88,6.01,0.00,63.0,1]])
```
