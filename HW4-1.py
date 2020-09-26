import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np

def get_list(n,acc,seeds):
    ''' data generation function
    n is number of sample
    acc is as like as variance 
    seeds is the random pack seed'''
    from random import random
    from random import seed
    from math import sqrt
    seed(seeds)
    X=list()
    Y=list()
    xm=list()
    ym=list()
    xp=list()
    yp=list()
    for i in range(0,n):
        x=random()
        y=sqrt( 1 - (x**2) )    
        signx=int( random()*2 )
        x=(-1)*x if signx==1 else x    
        signy=int( random()*2 )
        y=(-1)*y if signy==1 else y    
        sign_acc=int( random()*2 )
        A=(-1)*acc if sign_acc==1 else acc
        X.append(x+random()*A)    
        sign_acc=int( random()*2 )
        A=(-1)*acc if sign_acc==1 else acc
        Y.append(y+random()*A)    
        if (X[i]**2+Y[i]**2)<1 :
            xm.append(X[i])
            ym.append(Y[i])
        else:
            xp.append(X[i])
            yp.append(Y[i])  
    return xp,yp,xm,ym

#prepare test train
    
#train_DATA
n=100 #number of train
xp,yp,xm,ym=get_list(n,0.2,4)
xx=xp+xm
yy=yp+ym
Xl=list()
for i in range(n):
    Xl.append([xx[i],yy[i]])    
X = np.asarray(Xl, dtype=np.float32)
y1=[1 for x in range(len(xp))] 
y2=[-1 for x in range(len(xm))] 
y= np.asarray(y1+y2, dtype=np.integer)

#train_DATA
T=10 #number of train
xp,yp,xm,ym=get_list(n,0.2,4)
xx=xp+xm
yy=yp+ym
Xl=list()
for i in range(n):
    Xl.append([xx[i],yy[i]])    
X_test = np.asarray(Xl, dtype=np.float32)
y1=[1 for x in range(len(xp))] 
y2=[-1 for x in range(len(xm))] 
y_test= np.asarray(y1+y2, dtype=np.integer)


#ploting
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
circ = plt.Circle((0, 0), radius=1, edgecolor='Y', facecolor='None')
ax1.add_patch(circ)
plt.plot(xp,yp,'bo',xm,ym,'ro')
plt.axis("equal")
plt.show()

from sklearn import svm
from sklearn.model_selection import cross_val_score
#1-a___________________
clf1 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',
              coef0=0.0, shrinking=True, probability=False, tol=0.001,
              cache_size=200, class_weight=None, verbose=False, max_iter=-1,
              decision_function_shape='ovr', random_state=None)
clf1.fit(X,y) 
scores = cross_val_score(clf1, X , y , cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
plot_decision_regions(X=X, y=y, clf=clf1, legend=2)
plt.title('SVM Linear', size=16)
plt.axis("equal")
plt.show()
#1-b_________________
clf2 = svm.LinearSVC(penalty='l2', loss='squared_hinge',
                    dual=True, tol=0.00001, C=1.0, multi_class='crammer_singer',
                    fit_intercept=False, intercept_scaling=2,
                    class_weight=None, verbose=0, random_state=0,
                    max_iter=2000)

clf2.fit(X,y) 
scores = cross_val_score(clf2, X , y , cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
plot_decision_regions(X=X, y=y, clf=clf2, legend=2)
plt.title('soft SVM Linear', size=16)
plt.axis("equal")
plt.show()
#1-c_______________
clf3 = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
              coef0=0.0, shrinking=True, probability=False, tol=0.001,
              cache_size=200, class_weight=None, verbose=False, max_iter=-1,
              decision_function_shape='ovr', random_state=None)

clf3.fit(X,y) 
scores = cross_val_score(clf3, X , y , cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
plot_decision_regions(X=X, y=y, clf=clf3, legend=2)
plt.title('rbf, SVM ', size=16)
plt.axis("equal")
plt.show()
#1-d_______________
clf4 = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto',
              coef0=0.0, shrinking=True, probability=False, tol=0.001,
              cache_size=200, class_weight=None, verbose=False, max_iter=-1,
              decision_function_shape='ovr', random_state=None)

clf4.fit(X,y) 
scores = cross_val_score(clf4, X , y , cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
plot_decision_regions(X=X, y=y, clf=clf4, legend=2)
plt.title('poly, SVM ', size=16)
plt.axis("equal")
plt.show()
#1-e_______________
clf5 = svm.SVC(C=1.0, kernel='sigmoid', degree=5, gamma='auto',
              coef0=0.0, shrinking=True, probability=False, tol=0.001,
              cache_size=200, class_weight=None, verbose=False, max_iter=-1,
              decision_function_shape='ovr', random_state=None)

clf5.fit(X,y) 
scores = cross_val_score(clf5, X , y , cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
plot_decision_regions(X=X, y=y, clf=clf5, legend=2)
plt.title('sigmoid, SVM ', size=16)
plt.axis("equal")
plt.show()


