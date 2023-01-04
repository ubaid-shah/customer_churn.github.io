import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

train=pd.read_csv('Bank_churn.csv')

# converting gender to 0 and 1
# drop non numeric columns

Gender=pd.get_dummies(train['Gender'],drop_first=False)
train=pd.concat([train,Gender], axis=1)
train.drop(['RowNumber','CustomerId','Surname','Geography','Gender'], axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(train.drop('Exited',axis=1),
                                                    train['Exited'], test_size=0.2,
                                                    random_state=101)
# fit the model on the training variables X_train and Y_train

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

pickle.dump(logmodel, open('churn.pkl', 'wb'))
