import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


def read_csv(test, train):
    test_data = pd.read_csv(test)
    train_data = pd.read_csv(train)
    return test_data, train_data

test_data, train_data = read_csv('test.csv', 'train.csv')


# drop where target is missing
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
iowa_target = train_data.SalePrice


# find cols with missing data to drop
cols_with_missing = [col for col in train_data.columns
                                if train_data[col].isnull().any()]

# drop cols [...] + cols with missing data
iowa_train_predictors = train_data.drop(['Id','SalePrice'] + cols_with_missing,
                                        axis=1).select_dtypes(exclude=['object'])

iowa_test_predictors = test_data.drop(['Id'] + cols_with_missing,axis=1)


# select categorical columns with 'cardinality' 
low_card_cols = [cname for cname in iowa_train_predictors.columns if
                                    iowa_train_predictors[cname].nunique() <10 and
                                    iowa_train_predictors[cname].dtype=='object']
numeric_cols = [cname for cname in iowa_train_predictors.columns if 
                                iowa_train_predictors[cname].dtype in
                                ['int64','float64']] 

my_cols = low_card_cols + numeric_cols 
train_predictors = iowa_train_predictors[my_cols]
test_predictors = iowa_test_predictors[my_cols]


# one hot encode categorical 
one_hot_encoded_training_predictiors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictiors = pd.get_dummies(test_predictors)


# join the one_hot training predictors with the test predictors
final_train, final_test = one_hot_encoded_training_predictiors.align(
                                            one_hot_encoded_test_predictiors,
                                            join = 'inner',
                                            axis = 1)


train_X, test_X, train_y, test_y = train_test_split(final_train.values,
                                    iowa_target.values, test_size=0.25)

# create a pipeline for ease of use
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=10000,
                                                    learning_rate=0.01,
                                                    xgbregressor__early_stopping_rounds = 80, 
                                                    xgbregressor__eval_set=[(test_X, test_y)],
                                                    xgbregressor__verbose=False))
#train pipeline
my_pipeline.fit(train_X,train_y)


#----------- For Submission Purposes -----------#

# pull the one-hot-coded test data from above and impute it before prediction
my_imputer = Imputer()
final_imputed_test = pd.DataFrame(my_imputer.fit_transform(final_test))
final_imputed_test.columns = final_test.columns


# make prediction
predictions = my_pipeline.predict(final_imputed_test)
print(len(predictions))
print(predictions)

submission = pd.DataFrame({'Id' : test_data.Id, 'SalePrice': predictions})
submission.to_csv('submission.csv', index=False)

