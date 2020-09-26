import pandas as pd
from string import ascii_lowercase


def transformer (dataframe):
    import numpy as np
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(list(ascii_lowercase))
    a=dataframe.values.tolist()
    x=list()
    for i in range(len(a)):
        x.append(le.transform(a[i]))
        X=np.asarray(x, dtype=np.integer)
    return X

def test(X,Y,clf):
    A=list()
    x=list(clf.predict(X))
    y=list(Y)
    for i in range(len(X)):
        if(x[i]==y[i]):
            A.append(1)
        else:
            A.append(0)
    return sum(A)/len(A)

    
path='F:/Data/noisy_train.csv'
df_shroom=pd.read_csv(path)

path1='F:/Data/noisy_valid.csv'
df_shroom_valid=pd.read_csv(path1)

path2='F:/Data/noisy_test.csv'
df_shroom_test=pd.read_csv(path2)

from sklearn import svm
colmns=list(df_shroom.columns)
colmns.remove('poisonous')
#prepare data

X_train=df_shroom[colmns]
y_train=df_shroom['poisonous']

Xt_train=transformer(X_train)

X_valid=df_shroom_valid[colmns]
y_valid=df_shroom_valid['poisonous']

Xt_valid=transformer(X_valid)

X_test=df_shroom_test[colmns]
y_test=df_shroom_test['poisonous']

Xt_test=transformer(X_test)
    
#start learning part
clf = svm.SVC(C=1.0, kernel='rbf', degree=4, gamma='auto',
                  coef0=0.0, shrinking=True, probability=False, tol=0.00001,
                  cache_size=100000, class_weight=None, verbose=False, max_iter=-1,
                  decision_function_shape='ovr', random_state=None)
clf.fit(Xt_train,y_train)
    
print('the acc is :' , test(Xt_test,y_test,clf))








