# Importing the basic libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Reading the dataset
df = pd.read_csv('weather.csv')

# Getting datatypes of columns in the DataFrame
print("\n", "*"*50)
print("\n\n-----Initial DataFrame State:\n\n")
df.info()
# Printing DataFrame dimensions
print("\n\n-----Initial DataFrame Dimensions:\n\n-----", df.shape)

# Checking for null values in the DataFrame
print("\n\n-----Checking for null values initially\n\n-----", df.isnull().sum())

# Deleting the rows with the dropna() function
df.dropna(axis = 0, inplace = True)

# Checking for null values again after dropping the rows with null values
print("\n\n-----Checking for null values after dropping null values\n\n-----", df.isnull().sum())

# Checking the dimensions after dropping the rows with null values
print("\n\n-----Checking DataFrame shape after dropping null values\n\n-----", df.shape)

# Finally, printing the first 5 rows of the DataFrame
print("\n\n-----First 5 rows of the DataFrame\n\n-----", df.head())

# Here, let's change the RainToday and RainTomorrow to numeric labels using
# the map() function so that we can utilize them for our Regression model.
# Printing the distribution before mapping
print("\n\n-----Checking 'RainToday' column values and their counts initially\n\n-----",
      df['RainToday'].value_counts())
# Mapping the values of the column to convert the categorical values to integer
map1_dict = {'No': 0,
             'Yes': 1}

df['RainToday'] = df['RainToday'].map(map1_dict)
# Printing the distribution after mapping
print("\n\n-----Checking 'RainToday' column values and their counts after mapping\n\n-----",
      df['RainToday'].value_counts())

# Printing the distribution before mapping
print("\n\n-----Checking 'RainTomorrow' column values and their counts initially\n\n-----",
      df['RainTomorrow'].value_counts())
# Mapping the values of the column to convert the categorical values to integer
map1_dict = {'No': 0,
             'Yes': 1}

df['RainTomorrow'] = df['RainTomorrow'].map(map1_dict)
# Printing the distribution after mapping
print("\n\n-----Checking 'RainTomorrow' column values and their counts after mapping\n\n-----",
      df['RainTomorrow'].value_counts())

# Feature Engineering
# Here, we have three more columns with categorial values.
# But this time, changing them to rogue values like 0, 1, 2... could lead to faulty model.
# Hence, let's perform one-hot encoding using the get_dummies() function for these columns.

# Creating a 'weather_dummies_df' DataFrame using the 'get_dummies()' function on the categorical columns
categorical_df = df.select_dtypes(include = ['object'])

weather_dummies_df = pd.get_dummies(categorical_df, dtype = int, drop_first = True)

print("\n\n-----Printing one-hot encoded values:\n\n-----",
      weather_dummies_df.head())

# Now, let's drop the original categorial columns as they are of no use from now on
df.drop(list(categorical_df.columns), axis = 1, inplace = True)

# Now, let's concat the original DataFrame and dummy DataFrame using 'concat()' function
# By doing this, we can sustain the data of the 3 columns that were lost while dropping,
# but now in numeric form
df = pd.concat([df, weather_dummies_df], axis = 1)
print("\n\n-----First five rows of the DataFrame:\n\n-----", df.head())

# Here, we can see that all the columns are of numeric types now
print("\n\n-----Printing DataFrame state again:")
df.info()

# Now, let's check the correlation of all the columns
# Calculating correlation coefficient for all columns of the DataFrame
corr_coef = df.corr()
# Using heatmap to observe correlations.
plt.figure(figsize = (50, 20), dpi = 96)
plt.title("Heatmap of correlation:")
sns.heatmap(corr_coef, annot = True, fmt='.1g')
plt.show()

# Let's proceed with model building now
# Importing the module
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Splitting dataset into training and testing data
features = list(df.columns)
features.remove('RainTomorrow')

X = df[features]
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Normalising the train and test DataFrames using the standard normalisation method.
# Defining the 'standard_scalar()' function for calculating Z-scores
def standard_scaler(series):
  if (series.std() != 0):
    new_series = (series - series.mean()) / series.std()
    return new_series
  return series
# Creating the DataFrames norm_X_train and norm_X_train
norm_X_train = X_train.apply(standard_scaler, axis = 0)
norm_X_test = X_test.apply(standard_scaler, axis = 0)
# Applying the 'standard_scalar()' on X_train on numeric columns using apply() function
# and getting the descriptive statistics of the normalised X_train
print("\n\n-----Normalized train dataset:\n\n-----", norm_X_train.describe())
# Applying the 'standard_scalar()' on X_test on numeric columns using apply() function
# and getting the descriptive statistics of the normalised X_test
print("\n\n-----Normalized test dataset:\n\n-----", norm_X_test.describe())

# Deploying the 'LogisticRegression' model using the 'fit()' function.
lg_clf = LogisticRegression()
lg_clf.fit(norm_X_train, y_train)
print("\n\n-----Accuracy score of the model:", lg_clf.score(norm_X_train, y_train))

# Making predictions on the train dataset by using the 'predict()' function.
lg_clf_train_pred = lg_clf.predict(norm_X_train)
print("\n\n-----Prediction results of train dataset:\n\n-----", lg_clf_train_pred)

# Displaying the results of confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
print("\n\n-----Confusion matrix:\n\n-----", confusion_matrix(y_train, lg_clf_train_pred))

# Here, we can see above that the model has predicted with almost 100% accuracy, hence our model is working well.
# As we saw above, the model has very less false-positives and false-negatives.
# Displaying the results of classification report for train dataset prediction
print("\n\n-----Classification report on train dataset prediction\n\n-----",
      classification_report(y_train, lg_clf_train_pred))

# Making predictions on the test dataset by using the 'predict()' function.
lg_clf_test_pred = lg_clf.predict(norm_X_test)
print("\n\n-----Prediction results of test dataset:\n\n-----", lg_clf_test_pred)

# Displaying the results of confusion_matrix
print("\n\n-----Confusion matrix:\n\n-----", confusion_matrix(y_test, lg_clf_test_pred))

# We can see that the predictions are almost 100% accurate on the test dataset too.
# Displaying the results of classification report for test dataset prediction
print("\n\n-----Classification report on test dataset prediction\n\n-----",
      classification_report(y_test, lg_clf_test_pred))
