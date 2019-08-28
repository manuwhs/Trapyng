
"""
####### Basic classifier's library used for initial analysis #####
"""

# Import common useful libraries
import numpy as np
import pandas as pd
import copy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import  make_scorer     # To make a scorer for the GridSearch.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


"""
All of this functions implement a different classifier that follows the same format:
    - It takes as input:
        
        - Xtrain, Ytrain: The input data and labels used to train the algorithm 
        in the form of matrices (Nsam x Nfeatures).
        
        - Xtest = None , Ytest = None: Optional analogous input data to compute
        a measure of generalization of the trained model.
        
        - verbose = 0: If set to 1, it will print the scores of the train and test
        datasets given as input.
        
    - It returns:
        - The trained model.
        
"""

def get_LogReg(Xtrain, Ytrain, Xtest = None , Ytest = None, verbose = 0):
        lr = LogisticRegression()
        lr.fit(Xtrain,Ytrain)
        
        if (verbose == 1):
            scores = np.empty((2))
            scores[0] = lr.score(Xtrain,Ytrain)
            print('Logistic Regression, train: {0:.02f}% '.format(scores[0]*100))
            
            if (type(Xtest) != type(None)):
                scores[1] = lr.score(Xtest,Ytest)
                print('Logistic Regression, test: {0:.02f}% '.format(scores[1]*100))
        return lr
        

def get_LDA(Xtrain, Ytrain, Xtest = None , Ytest = None, verbose = 0):
        lda = LDA()
        lda.fit(Xtrain,Ytrain)
        scores = np.empty((4))

        if (verbose == 1):
            scores = np.empty((2))
            
            scores[0] = lda.score(Xtrain,Ytrain)
            print('LDA, train: {0:.02f}% '.format(scores[0]*100))
            
            if (type(Xtest) != type(None)):
                scores[1] = lda.score(Xtest,Ytest)
                print('LDA, test: {0:.02f}% '.format(scores[1]*100))
        
        return lda

def get_QDA(Xtrain, Ytrain, Xtest = None , Ytest = None, verbose = 0):
    qda = QDA()
    qda.fit(Xtrain,Ytrain)
    
    scores = np.empty((2))
    if (verbose == 1):
        scores[0] = qda.score(Xtrain,Ytrain)
        print('QDA, train: {0:.02f}% '.format(scores[0]*100))
        if (type(Xtest) != type(None)):
            scores[1] = qda.score(Xtest,Ytest)
            print('QDA, test: {0:.02f}% '.format(scores[1]*100))
    return qda
    
def get_GNB(Xtrain, Ytrain, Xtest = None , Ytest = None, verbose = 0):
    gnb = GaussianNB()
    gnb.fit(Xtrain,Ytrain)
    
    if (verbose == 1):
        scores = np.empty((2))
        scores[0] = gnb.score(Xtrain,Ytrain)
        print('GNB, train: {0:.02f}% '.format(scores[0]*100))
        if (type(Xtest) != type(None)):
            scores[1] = gnb.score(Xtest,Ytest)
            print('GNB, test: {0:.02f}% '.format(scores[1]*100))
    return gnb
    
def get_LSVM(Xtrain, Ytrain, Xtest = None , Ytest = None, verbose = 0):
    C = np.logspace(-3,3,2)

    # Create dictionaries with the Variables for the validation !
    # We create the dictinary for every TYPE of SVM we are gonna use.
    param_grid_linear = dict()
    param_grid_linear.update({'kernel':['linear']})
    param_grid_linear.update({'C':C})
    
    # The folds of "StratifiedKFold" are made by preserving the percentage of samples for each class.
    
    stkfold = StratifiedKFold(Ytrain, n_folds = 5)
    # GridSearchCV implements a CV over a variety of Parameter values !! 
    # In this case, over C fo the linear case, C and "degree" for the poly case
    # and C and "gamma" for the rbf case. 
    # The parameters we have to give it are:
    # 1-> Classifier Object: SVM, LR, RBF... or any other one with methods .fit and .predict
    # 2 -> Subset of parameters to validate. C 
    # 3 -> Type of validation: K-fold
    # 4 -> Scoring function. sklearn.metrics.accuracy_score
    # The score function is the one we want to minimize or maximize given the label and the predicted.
    acc_scorer = make_scorer(accuracy_score)

    classifier =  SVC(class_weight='balanced',probability = True)
    gsvml = GridSearchCV(classifier,param_grid_linear, scoring = acc_scorer,
                         cv = stkfold, refit = True,n_jobs=-1)
    gsvml.fit(Xtrain,Ytrain)
    
    if (verbose == 1):
        scores = np.empty((4))
        scores[0] = gsvml.score(Xtrain,Ytrain)
        print('LSVM, train: {0:.02f}% '.format(scores[0]*100))
        
        if (type(Xtest) != type(None)):
            scores[1] = gsvml.score(Xtest,Ytest)
            print('LSVM, test: {0:.02f}% '.format(scores[1]*100))
            
    return gsvml


def get_SVM_poly(Xtrain, Ytrain, Xtest = None , Ytest = None, verbose = 0):
        # Parameters for the validation
        C = np.logspace(-3,3,20)
        p = np.arange(2,4)

        param_grid_pol = dict()
        param_grid_pol.update({'kernel':['poly']})
        param_grid_pol.update({'C':C})
        param_grid_pol.update({'degree':p})
        
        stkfold = StratifiedKFold(Ytrain, n_folds = 5)
        
        # The score function is the one we want to minimize or maximize given the label and the predicted.
        acc_scorer = make_scorer(accuracy_score)
        classifier =  SVC(class_weight='balanced',probability = True)
        gsvmr = GridSearchCV(classifier,param_grid_pol, scoring =acc_scorer,
                             cv = stkfold, refit = True,n_jobs=-1)

        gsvmr.fit(Xtrain,Ytrain)
        
        if (verbose == 1):
            scores = np.empty((4))
            scores[0] = gsvmr.score(Xtrain,Ytrain)
            print('SVM Poly, train: {0:.02f}% '.format(scores[0]*100))
            
            if (type(Xtest) != type(None)):
                scores[1] = gsvmr.score(Xtest,Ytest)
                print('SVM Poly, test: {0:.02f}% '.format(scores[1]*100))
        return gsvmr
    
def get_SVM_rf(Xtrain, Ytrain, Xtest = None , Ytest = None, verbose = 0):
        # Parameters for the validation
        C = np.logspace(-3,3,20)
        gamma = np.array([0.125,0.25,0.5,1,2,4])/200
        
        param_grid_rbf = dict()
        param_grid_rbf.update({'kernel':['rbf']})
        param_grid_rbf.update({'C':C})
        param_grid_rbf.update({'gamma':gamma})
        
        stkfold = StratifiedKFold(Ytrain, n_folds = 5)

        acc_scorer = make_scorer(accuracy_score)
        classifier =  SVC(class_weight='balanced',probability = True)
        gsvmr = GridSearchCV(classifier,param_grid_rbf, scoring =acc_scorer,
                             cv = stkfold, refit = True,n_jobs=-1)

        gsvmr.fit(Xtrain,Ytrain)
        
        if (verbose == 1):
            scores = np.empty((4))
            scores[0] = gsvmr.score(Xtrain,Ytrain)
            print('SVM_rf, train: {0:.02f}% '.format(scores[0]*100))
            
            if (type(Xtest) != type(None)):
                scores[1] = gsvmr.score(Xtest,Ytest)
                print('SVM_rf, test: {0:.02f}% '.format(scores[1]*100))
        return gsvmr
        
def get_KNN(Xtrain, Ytrain, Xtest = None , Ytest = None, verbose = 0):
    # Perform authomatic grid search
    params = [{'n_neighbors':np.arange(1,10)}]
    gknn = GridSearchCV(KNeighborsClassifier(),params,scoring='precision',
                        cv=4,refit=True,n_jobs=-1)
    gknn.fit(Xtrain,Ytrain)
    
    if (verbose == 1):
        scores = np.empty((4))
        scores[0] = gknn.score(Xtrain,Ytrain)
        print('{0}-NN, train: {1:.02f}% '.format(gknn.best_estimator_.n_neighbors,scores[0]*100))
        
        if (type(Xtest) != type(None)):
            scores[1] = gknn.score(Xtest,Ytest)
            print('{0}-NN, test: {1:.02f}% '.format(gknn.best_estimator_.n_neighbors,scores[1]*100))
    
    return gknn

"""
TREE BASED !!!!
"""

def get_TreeCl(Xtrain, Ytrain, Xtest = None , Ytest = None, verbose = 0):
   #%% Tree Classifier
    param_grid = dict()
    param_grid.update({'max_features':['auto',"log2", 'sqrt']})
    param_grid.update({'max_depth':np.arange(1,21)})
    param_grid.update({'min_samples_split':np.arange(2,11)})
    gtree = GridSearchCV(DecisionTreeClassifier(),param_grid,scoring='precision',
                         cv=5,refit=True,n_jobs=-1)
    gtree.fit(Xtrain,Ytrain)
    
    if (verbose == 1):
        scores = np.empty((2))
        scores[0] = gtree.score(Xtrain,Ytrain)
        print('Decision Tree, train: {0:.02f}% '.format(scores[0]*100))
        
        if (type(Xtest) != type(None)):
            scores[1] = gtree.score(Xtest,Ytest)
            print('Decision Tree, test: {0:.02f}% '.format(scores[1]*100))
    
    return gtree
    
def get_RF(Xtrain, Ytrain, baseTree, Xtest = None , Ytest = None, verbose = 0):
    # Random Forest
    rf = RandomForestClassifier(n_estimators=1000,max_features=baseTree.best_estimator_.max_features,
                                max_depth=baseTree.best_estimator_.max_depth,
                                min_samples_split=baseTree.best_estimator_.min_samples_split,oob_score=True,n_jobs=-1)
    rf.fit(Xtrain,Ytrain)
    
    if (verbose == 1):
        scores = np.empty((4))
        scores[0] = rf.score(Xtrain,Ytrain)
        print('Random Forest, train: {0:.02f}% '.format(scores[0]*100))
        
        if (type(Xtest) != type(None)):
            scores[1] = rf.score(Xtest,Ytest)
            print('Random Forest, test: {0:.02f}% '.format(scores[1]*100))

    return rf
    
def get_ERT(Xtrain, Ytrain, baseTree, Xtest = None , Ytest = None, verbose = 0):
    # Extremely Randomized Trees
    ert = ExtraTreesClassifier(n_estimators=1000,max_features=baseTree.best_estimator_.max_features,
                               max_depth=baseTree.best_estimator_.max_depth,
                               min_samples_split=baseTree.best_estimator_.min_samples_split,n_jobs=-1)
    ert.fit(Xtrain,Ytrain)
    
    if (verbose == 1):
        scores = np.empty((2))
        scores[0] = ert.score(Xtrain,Ytrain)
        print('Extremely Randomized Trees, train: {0:.02f}% '.format(scores[0]*100))
        if (type(Xtest) != type(None)):
            scores[1] = ert.score(Xtest,Ytest)
            print('Extremely Randomized Trees, test: {0:.02f}% '.format(scores[1]*100))

    return ert
        

    
