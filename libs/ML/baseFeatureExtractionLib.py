import numpy as np
import matplotlib.pyplot as plt
import Utilities as util
import timeit

from sklearn.decomposition import PCA,KernelPCA
from sklearn.cross_decomposition import PLSSVD,PLSCanonical,PLSRegression
from sklearn.cross_decomposition import CCA
from sklearn.lda import LDA
import sklearn.metrics.pairwise as pair
from sklearn.preprocessing import KernelCenterer
from sklearn.kernel_approximation import Nystroem,RBFSampler

plt.close('all')

#%% Load data
Load_Data = 1;
Data_Prep = 1;

if (Load_Data == 1):
    data = np.loadtxt("AVIRIS_dataset/data.txt")
    labels = np.loadtxt("AVIRIS_dataset/labels.txt")
    names = np.loadtxt("AVIRIS_dataset/names.txt", dtype=np.str)


#################################################################
#################### DATA PREPROCESSING #########################
#################################################################


if (Data_Prep == 1):
    #%% Remove noisy bands
    dataR1 = data[:,:103]
    dataR2 = data[:,108:149]
    dataR3 = data[:,163:219]
    dataR = np.concatenate((dataR1,dataR2,dataR3),axis=1)
    
    #%% Exclude background class
    dataR = dataR[labels!=0,:]
    labelsR = labels[labels!=0]
    labelsR = labelsR - 1  # So that classes start at 1
    #%% Split data in training and test sets
    train_ratio = 0.2
    rang = np.arange(np.shape(dataR)[0],dtype=int) # Create array of index
    np.random.seed(0)
    rang = np.random.permutation(rang)        # Randomize the array of index
    
    Ntrain = round(train_ratio*np.shape(dataR)[0])    # Number of samples used for training
    Ntest = len(rang)-Ntrain                  # Number of samples used for testing
    Xtrain = dataR[rang[:Ntrain]]
    Xtest = dataR[rang[Ntrain:]]
    Ytrain = labelsR[rang[:Ntrain]]
    Ytest = labelsR[rang[Ntrain:]]
    
    labelsTrain = labelsR[rang[:Ntrain]]
    labelsTest = labelsR[rang[Ntrain:]]
    
    #%% Normalize data
    mx = np.mean(Xtrain,axis=0,dtype=np.float64)
    stdx = np.std(Xtrain,axis=0,dtype=np.float64)
    
    Xtrain = np.divide(Xtrain-np.tile(mx,[Ntrain,1]),np.tile(stdx,[Ntrain,1]))
    Xtest = np.divide(Xtest-np.tile(mx,[Ntest,1]),np.tile(stdx,[Ntest,1]))
    
    #==============================================================================
    # # Also we could have used:
    # from sklearn import preprocessing
    # scaler = preprocessing.StandardScaler().fit(Xtrain)
    # Xtrain = scaler.transform(Xtrain)            
    # Xtest = scaler.transform(Xtest)       
    #==============================================================================

    nClasses = np.alen(np.unique(labelsR))
    nFeatures = np.shape(dataR)[1]    # Create output coding matrix Y
    
    Ytrain = np.zeros((Ntrain,np.alen(np.unique(labelsTrain))))
    for i in np.unique(labelsTrain):
        Ytrain[labelsTrain==i,i-1] = True
    Ytest = np.zeros((Ntest,np.alen(np.unique(labelsTest))))
    for i in np.unique(labelsTest):
        Ytest[labelsTest==i,i-1] = True
    
    # Create a set of random colors for each class
    np.random.seed()
    classColors = np.random.rand(nClasses,3)
    
    # Global variables
    nComponents = np.arange(1,nFeatures,8)

### LINEAR METHODS ###
#%% PRINCIPAL COMPONENT ANALYSIS
# Plot explained variance vs number of components

if (0):
    pca = PCA()
    pca.fit_transform(Xtrain)
    
    cumvarPCA = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot classification score vs number of components
    nComponents = np.arange(1,nFeatures,8)
    pcaScores = np.zeros((5,np.alen(nComponents)))
    
    for i,n in enumerate(nComponents):   
        pca = PCA(n_components=n,whiten=False)
        XtrainT = pca.fit_transform(Xtrain)
        XtestT = pca.transform(Xtest)
        pcaScores[:,i] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)

    
    #%% Plot figures
    #%% Plot explained variance
    plt.figure()
    plt.plot(np.arange(1,np.alen(cumvarPCA)+1),cumvarPCA,lw=3,label='PCA')
    plt.xlim(1,np.alen(cumvarPCA))
    plt.legend(loc='lower right')
    plt.title('Explained Variance Ratio')
    plt.xlabel('Number of components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.show()
    
    #%% Plot accuracies for PCA 
    nComponents = np.arange(1,nFeatures,8)
    plt.figure()
    plt.plot(nComponents,pcaScores[0,:],lw=3,label='LR')
    plt.plot(nComponents,pcaScores[1,:],lw=3,label='LDA')
    plt.plot(nComponents,pcaScores[2,:],lw=3,label='GNB')
    plt.plot(nComponents,pcaScores[3,:],lw=3,label='Linear SVM')
    plt.plot(nComponents,pcaScores[4,:],lw=3,label='rbf SVM')

    plt.xlim(1,np.amax(nComponents))
    plt.title('PCA accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.legend (['LR','LDA','GNB','Linear SVM','rbf SVM'],loc='lower right')
    plt.grid(True)

if (1):
 #%% Plot data proyections for PCA

    pca = PCA(n_components=2)
    xtPCA = pca.fit_transform(Xtrain)
    uPCA = pca.components_
    
    # View the  original data and the projection vectors
    labels = np.unique(labelsTrain.astype(np.int))
    plt.figure()
    for i,l in enumerate(labels):
        plt.scatter(xtPCA[labelsTrain==l,0],xtPCA[labelsTrain==l,1],alpha=0.5,c=classColors[i,:])
    plt.title('First 2 components of projected data')
    plt.show()

  ######################################
  # Only left with 2 components to show the cahnge of PCA
    # Just to show the hyperplane of maximum varinace in 2 D, we transform the data into 2 D.
    
    # Training data with 2 dimensions
    pca = PCA(n_components=2)
    Xtrain_2d = Xtrain[:,:2]
    xtPCA = pca.fit_transform(Xtrain_2d)
    uPCA = pca.components_
        
    plt.figure()
    for i,l in enumerate(labels):
        plt.scatter(Xtrain_2d[labelsTrain==l,0],Xtrain_2d[labelsTrain==l,1],alpha=0.5,c=classColors[i,:])
    
    plt.quiver(uPCA[0,0],uPCA[0,1],color='k',edgecolor='k',lw=1,scale=5)
    plt.quiver(uPCA[1,0],uPCA[1,1],color='k',edgecolor='k',lw=1,scale=10)
    plt.title('Original Data and first 2 eigenvectors')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.show()

    plt.figure()
    for i,l in enumerate(labels):
        plt.scatter(xtPCA[labelsTrain==l,0],xtPCA[labelsTrain==l,1],alpha=0.5,c=classColors[i,:])

    plt.title('Projected data over the components')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.show()

if (0):
    
    #%% PARTIAL LEAST SQUARES
    #%% PLS SVD
    nComponents = np.arange(1,nClasses+1)
    plsSvdScores = np.zeros((5,np.alen(nComponents)))
    for i,n in enumerate(nComponents):
        plssvd = PLSSVD(n_components=n)
        plssvd.fit(Xtrain,Ytrain)
        XtrainT = plssvd.transform(Xtrain)
        XtestT = plssvd.transform(Xtest)
        plsSvdScores[:,i] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)
        
    plssvd = PLSSVD(n_components=2)
    xt,yt = plssvd.fit_transform(Xtrain,Ytrain)
    fig = plt.figure()
    util.plotData(fig,xt,labelsTrain,classColors)
    plt.title('First 2 components of projected data')
    
    #%% Plot accuracies for PLSSVD 
    plt.figure()
    for i in range (5):
        plt.plot(nComponents,plsSvdScores[i,:],lw=3)

    plt.xlim(1,np.amax(nComponents))
    plt.title('PLS SVD accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.legend (['LR','LDA','GNB','Linear SVM','rbf SVM'],loc='lower right')
    plt.grid(True)

if (0):
    #%% PLS Cannonical
    nComponents = np.arange(1,nClasses+1)
    plsCanScores = np.zeros((5,np.alen(nComponents)))
    for i,n in enumerate(nComponents):
        plscan = PLSCanonical(n_components=n)
        plscan.fit(Xtrain,Ytrain)
        XtrainT = plscan.transform(Xtrain)
        XtestT = plscan.transform(Xtest)
        plsCanScores[:,i] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)

    
    plscan = PLSCanonical(n_components=2)
    plscan.fit(Xtrain,Ytrain)
    xt = plscan.transform(Xtrain)
    fig = plt.figure()
    util.plotData(fig,xt,labelsTrain,classColors)
    plt.title('First 2 components of projected data')
    

    #%% Plot accuracies for PLSSVD 
    plt.figure()
    for i in range (5):
        plt.plot(nComponents,plsCanScores[i,:],lw=3)

    plt.xlim(1,np.amax(nComponents))
    plt.title('PLS Cannonical accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.legend (['LR','LDA','GNB','Linear SVM','rbf SVM'],loc='lower right')
    plt.grid(True)

if (0):
    #%% PLS Regression
    nComponents = np.arange(1,nClasses+1)
    plsRegScores = np.zeros((5,np.alen(nComponents)))
    for i,n in enumerate(nComponents):
        plsReg = PLSRegression(n_components=n)
        plsReg.fit(Xtrain,Ytrain)
        XtrainT = plsReg.transform(Xtrain)
        XtestT = plsReg.transform(Xtest)
        plsRegScores[:,i] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)

    
    plsReg = PLSRegression(n_components=2)
    plsReg.fit(Xtrain,Ytrain)
    xt = plsReg.transform(Xtrain)
    fig = plt.figure()
    util.plotData(fig,xt,labelsTrain,classColors)
    plt.title('First 2 components of projected data')
    

    #%% Plot accuracies for PLSSVD 
    plt.figure()
    for i in range (5):
        plt.plot(nComponents,plsRegScores[i,:],lw=3)

    plt.xlim(1,np.amax(nComponents))
    plt.title('PLS Regression accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.legend (['LR','LDA','GNB','Linear SVM','rbf SVM'],loc='lower right')
    plt.grid(True)

if (0):
    #%% Canonical Correlation Analysis
    nComponents = np.arange(1,nClasses +1)
    cca = CCA(n_components=nClasses)
    cca.fit(Xtrain,Ytrain)
    XtrainT = cca.transform(Xtrain)
    XtestT = cca.transform(Xtest)
    ccaScores = np.zeros((5,np.alen(nComponents)))
    for i,n in enumerate(nComponents):
        ccaScores[:,i] = util.classify(XtrainT[:,0:n],XtestT[:,0:n],labelsTrain,labelsTest)
    
    cca = CCA(n_components=3)
    cca.fit(Xtrain,Ytrain)
    xt = cca.transform(Xtrain)
    fig = plt.figure()
    util.plotData(fig,xt,labelsTrain,classColors)
    plt.title('First 3 components of projected data')
    

    #%% Plot accuracies for CCA
    plt.figure()
    for i in range (5):
        plt.plot(nComponents,ccaScores[i,:],lw=3)

    plt.xlim(1,np.amax(nComponents))
    plt.title('CCA accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.legend (['LR','LDA','GNB','Linear SVM','rbf SVM'],loc='lower right')
    plt.grid(True)

if (0):
    #%% Linear Discriminant Analysis
    nComponents = np.arange(1,nClasses+1)
    ldaScores = np.zeros((5,np.alen(nComponents)))
    for i,n in enumerate(nComponents):
        ldaT = LDA(n_components=n)
        ldaT.fit(Xtrain,labelsTrain)
        XtrainT = ldaT.transform(Xtrain)
        XtestT = ldaT.transform(Xtest)
        ldaScores[:,i] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)
        
    lda = LDA(n_components=3)
    lda.fit(Xtrain,labelsTrain)
    xt = lda.transform(Xtrain)
    fig = plt.figure()
    util.plotData(fig,xt,labelsTrain,classColors)
    plt.title('First 3 components of projected data')
    
    #%% Plot accuracies for LDA
    plt.figure()
    for i in range (5):
        plt.plot(nComponents,ldaScores[i,:],lw=3)

    plt.xlim(1,np.amax(nComponents))
    plt.title('LDA accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.legend (['LR','LDA','GNB','Linear SVM','rbf SVM'],loc='lower right')
    plt.grid(True)

if (0):
    # ICA
    from sklearn.decomposition import FastICA

    nComponents = np.arange(1,nClasses+1 + 50)
    icaScores = np.zeros((5,np.alen(nComponents)))
    for i,n in enumerate(nComponents):
        icaT = FastICA(n_components=n, max_iter = 10000)
        icaT.fit(Xtrain,labelsTrain)
        XtrainT = icaT.transform(Xtrain)
        XtestT = icaT.transform(Xtest)
        icaScores[:,i] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)
        
    ica = FastICA(n_components=3, max_iter = 10000)
    ica.fit(Xtrain,labelsTrain)
    xt = ica.transform(Xtrain)
    fig = plt.figure()
    util.plotData(fig,xt[:,:3],labelsTrain,classColors)
    plt.title('First 3 components of projected data')
    
    #%% Plot accuracies for ICA
    plt.figure()
    for i in range (5):
        plt.plot(nComponents,icaScores[i,:],lw=3)

    plt.xlim(1,np.amax(nComponents))
    plt.title('ICA accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.legend (['LR','LDA','GNB','Linear SVM','rbf SVM'],loc='lower right')
    plt.grid(True)       

if (0):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NONLINEAR METHODS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NONLINEAR METHODS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NONLINEAR METHODS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
        
    d = pair.pairwise_distances(Xtrain,Xtrain)
    aux = np.triu(d)
    sigma = np.sqrt(np.mean(np.power(aux[aux!=0],2)*0.5))
    gamma = 1/(2*sigma**2)

if (0):
    #%% K-PCA
    # Calculate accumulated variance
    kpca = KernelPCA(kernel="rbf",gamma=gamma)
    kpca.fit_transform(Xtrain)
    eigenvals = kpca.lambdas_[0:220]

    
    # Calculate classifiation scores for each component
    nComponents =  np.linspace(1, 500, 100, endpoint=True)
    kpcaScores = np.zeros((5,np.alen(nComponents)))
    
    kpca = KernelPCA(n_components = Ntrain,kernel="rbf",gamma=gamma)
    kpca.fit(Xtrain)
    XtrainT = kpca.transform(Xtrain)
    XtestT = kpca.transform(Xtest)
    

    for i in range(len(nComponents)):   
        kpcaScores[:,i] = util.classify(XtrainT[:,:nComponents[i]],XtestT[:,:nComponents[i]],labelsTrain,labelsTest)

    #%% Plot accuracies for kPCA
    plt.figure()
    for i in range (5):
        plt.plot(nComponents,kpcaScores[i,:],lw=3)

    plt.xlim(1,np.amax(nComponents))
    plt.title('kPCA accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.legend (['LR','LDA','GNB','Linear SVM','rbf SVM'],loc='lower right')
    plt.grid(True)      

if (0):

    # Calculate classifiation scores for each component
    nComponents =  np.linspace(500, 1500, 100, endpoint=True)
    kpcaldaScores = np.zeros((np.alen(nComponents),1))
    lda = LDA()

    for i in range(len(nComponents)):   
        lda.fit(XtrainT[:,:nComponents[i]],labelsTrain)
        kpcaldaScores[i] = lda.score(XtestT[:,:nComponents[i]],labelsTest)

#    %% Plot accuracies for kPCA
    plt.figure()
    plt.plot(nComponents,kpcaldaScores,lw=3)

    plt.xlim(1,np.amax(nComponents))
    plt.title('kPCA accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.xlim([500,1500])
    plt.legend (['LDA'],loc='lower right')
    plt.grid(True)    

if(0):
    # K-PCA second round
    ktrain = pair.rbf_kernel(Xtrain,Xtrain,gamma)
    ktest = pair.rbf_kernel(Xtest,Xtrain,gamma)
    kcent = KernelCenterer()
    kcent.fit(ktrain)
    ktrain = kcent.transform(ktrain)
    ktest = kcent.transform(ktest)
    
    kpca = PCA()
    kpca.fit_transform(ktrain)
    cumvarkPCA2 = np.cumsum(kpca.explained_variance_ratio_[0:220])
    
    # Calculate classifiation scores for each component
    nComponents = np.arange(1,nFeatures)
    kpcaScores2 = np.zeros((5,np.alen(nComponents)))
    for i,n in enumerate(nComponents):   
        kpca2 = PCA(n_components=n)
        kpca2.fit(ktrain)
        XtrainT = kpca2.transform(ktrain)
        XtestT = kpca2.transform(ktest)
        kpcaScores2[:,i] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)

    #%% Plot accuracies for kPCA
    plt.figure()
    for i in range (5):
        plt.plot(nComponents,kpcaScores2[i,:],lw=3)

    plt.xlim(1,np.amax(nComponents))
    plt.title('kPCA2 accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.legend (['LR','LDA','GNB','Linear SVM','rbf SVM'],loc='lower right')
    plt.grid(True)   
    
if (0):
    #%% K-PLS
#    ktrain = pair.rbf_kernel(Xtrain,Xtrain,gamma)
#    ktest = pair.rbf_kernel(Xtest,Xtrain,gamma)
#    kcent = KernelCenterer()
#    kcent.fit(ktrain)
#    ktrain = kcent.transform(ktrain)
#    ktest = kcent.transform(ktest)
    
    # Calculate classifiation scores for each component
    nComponents =  np.linspace(1, 200, 50, endpoint=True)
    kplsScores = np.zeros((5,np.alen(nComponents)))
    
    kpls = PLSRegression(n_components = 200)
    kpls.fit(ktrain,Ytrain)
    XtrainT = kpls.transform(ktrain)
    XtestT = kpls.transform(ktest)
    

    for i in range(len(nComponents)):   
        kplsScores[:,i] = util.classify(XtrainT[:,:nComponents[i]],XtestT[:,:nComponents[i]],labelsTrain,labelsTest)

    #%% Plot accuracies for kPCA
    plt.figure()
    for i in range (5):
        plt.plot(nComponents,kplsScores[i,:],lw=3)

    plt.xlim(1,np.amax(nComponents))
    plt.title('kPLS Regression accuracy')
    plt.xlabel('Number of components')
    plt.ylabel('accuracy')
    plt.legend (['LR','LDA','GNB','Linear SVM','rbf SVM'],loc='lower right')
    plt.grid(True)      




if(0):
    #%% K-CCA
    ktrain = pair.rbf_kernel(Xtrain,Xtrain,gamma)
    ktest = pair.rbf_kernel(Xtest,Xtrain,gamma)
    kcent = KernelCenterer()
    kcent.fit(ktrain)
    ktrain = kcent.transform(ktrain)
    ktest = kcent.transform(ktest)
    
    nComponents = np.arange(1,nClasses+1)
    kcca = CCA(n_components=nClasses)
    kcca.fit(ktrain,Ytrain)
    XtrainT = kcca.transform(ktrain)
    XtestT = kcca.transform(ktest)
    kccaScores = np.zeros((2,np.alen(nComponents)))
    for i,n in enumerate(nComponents):   
        kccaScores[:,i] = util.classify(XtrainT[:,0:n],XtestT[:,0:n],labelsTrain,labelsTest)
    
    #%% Subsampling methods
    kpls = PLSRegression(n_components=150)
    nComponents = np.arange(173,2173,100)
    
    # Nystroem method
    elapTimeNys = np.zeros(np.shape(nComponents))
    kplsScoresNys = np.zeros((2,3))
    for i,n in enumerate(nComponents):
        nys = Nystroem(n_components=n,gamma=gamma)
        nys.fit(Xtrain)
        ktrain = nys.transform(Xtrain)
        ktest = nys.transform(Xtest)
        startTime = timeit.default_timer()
        kpls.fit(ktrain,Ytrain)
        elapTimeNys[i] = timeit.default_timer() - startTime
        XtrainT = kpls.transform(ktrain)
        XtestT = kpls.transform(ktest)
        
        if n==573:
            kplsScoresNys[:,0] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)
        elif n==1073:
            kplsScoresNys[:,1] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)
        elif n==1573:
            kplsScoresNys[:,2] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)
    
    # RBF sampler method
    elapTimeRBFS = np.zeros(np.shape(nComponents))
    kplsScoresRBFS = np.zeros((2,3))
    for i,n in enumerate(nComponents):
        rbfs = RBFSampler(n_components=n,gamma=gamma)
        rbfs.fit(Xtrain)
        ktrain = rbfs.transform(Xtrain)
        ktest = rbfs.transform(Xtest)
        startTime = timeit.default_timer()
        kpls.fit(ktrain,Ytrain)
        elapTimeRBFS[i] = timeit.default_timer() - startTime
        XtrainT = kpls.transform(ktrain)
        XtestT = kpls.transform(ktest)
        
        if n==573:
            kplsScoresRBFS[:,0] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)
        elif n==1073:
            kplsScoresRBFS[:,1] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)
        elif n==1573:
            kplsScoresRBFS[:,2] = util.classify(XtrainT,XtestT,labelsTrain,labelsTest)
            
    #%% Plot figures
    #%% Plot explained variance vs number of components for PCA and kPCA
    plt.figure()
    plt.plot(np.arange(1,np.alen(cumvarPCA)+1),cumvarPCA,c='c',lw=2,label='Linear PCA')
    plt.plot(np.arange(1,np.alen(cumvarkPCA2)+1),cumvarkPCA2,c='r',lw=2,label='Gaussian kernel PCA')
    plt.xlim(1,np.alen(cumvarPCA))
    plt.legend(loc='lower right')
    plt.title('Explained Variance Ratio')
    plt.xlabel('number of components')
    plt.ylabel('explained variance ratio')
    plt.grid(True)
    plt.show()
    
    #%% Plot accuracies for PCA vs kPCA
    nComponents = np.arange(1,nFeatures,8)
    plt.figure()
    plt.plot(nComponents,pcaScores[0,:],'c',lw=2,label='Linear SVM for PCA')
    plt.plot(nComponents,pcaScores[1,:],'r',lw=2,label='RBF SVM for PCA')
    plt.plot(nComponents,kpcaScores2[0,:],'c--',lw=2,label='Linear SVM for kernelPCA')
    plt.plot(nComponents,kpcaScores2[1,:],'r--',lw=2,label='RBF SVM for kernelPCA')
    plt.xlim(1,np.amax(nComponents))
    plt.title('PCA and kernelPCA classification performance')
    plt.xlabel('number of components')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    #%% Plot data proyections for PCA
    labels = np.unique(labelsTrain.astype(np.int))
    plt.figure()
    for i,l in enumerate(labels):
        plt.scatter(xtPCA[labelsTrain==l,0],xtPCA[labelsTrain==l,1],alpha=0.5,c=classColors[i,:])
    plt.quiver(uPCA[0,0],uPCA[1,0],color='k',edgecolor='k',lw=1,scale=0.2)
    plt.quiver(-uPCA[1,0],uPCA[0,0],color='k',edgecolor='k',lw=1,scale=0.6)
    plt.title('2 dimensional PCA training data')
    plt.show()
    
    #%% Plot data proyections for PLS2
    plt.figure()
    for i,l in enumerate(labels):
        plt.scatter(xtPLS[labelsTrain==l,0],xtPLS[labelsTrain==l,1],alpha=0.5,c=classColors[i,:])
    plt.quiver(uPLS[0,0],uPLS[1,0],color='k',edgecolor='k',lw=1,scale=0.3)
    plt.quiver(-uPLS[1,0],uPLS[0,0],color='k',edgecolor='k',lw=1,scale=0.6)
    plt.title('2 dimensional PLS2 training data')
    plt.show()
    
    #%% Plot accuracies for linear methods
    nComponents = np.arange(1,nFeatures,8)
    nComponents2 = np.arange(1,nClasses+1)
    plt.figure()
    plt.plot(nComponents,pcaScores[0,:],'c',lw=2,label='Linear SVM for PCA')
    plt.plot(nComponents,pls2Scores[0,:],'r',lw=2,label='Linear SVM for PLS2')
    plt.plot(nComponents2,ccaScores[0,:],'y',lw=2,label='Linear SVM for CCA')
    plt.plot(nComponents2,ldaScores[0,:],'k',lw=2,label='Linear SVM for LDA')
    plt.xlim(1,np.amax(nComponents))
    plt.title('Comparison between linear methods')
    plt.xlabel('number of components')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    #%% Plot accuracies for nonlinear methods
    nComponents = np.arange(1,nFeatures,8)
    plt.figure()
    plt.plot(nComponents,kpcaScores2[0,:],'c',lw=2,label='Linear SVM for PCA')
    plt.plot(nComponents,kplsScores[0,:],'r',lw=2,label='Linear SVM for PLS2')
    #plt.plot(nComponents,kccaScores[0,:],'y',lw=2,label='Linear SVM for CCA')
    plt.xlim(1,np.amax(nComponents))
    plt.title('Comparison between nonlinear methods')
    plt.xlabel('number of components')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    #%% Plot execution time for kernel PLS2 and subsampled methods
    nComponents = np.arange(173,2173,100)
    plt.figure()
    plt.plot(nComponents,np.tile(elapTime,(20)),'k',lw=2,label='Full kernel')
    plt.plot(nComponents,elapTimeNys,'r',lw=2,label='Nystroem method')
    plt.plot(nComponents,elapTimeRBFS,'y',lw=2,label='RBF sampler')
    plt.xlim(173,np.amax(nComponents))
    plt.title('Execution time comparison for full and compacted kernels')
    plt.xlabel('number of kernel components')
    plt.ylabel('time units')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    #%% Plot accuracies for some number of components with linear SVM
    plt.figure()
    plt.bar(2,kplsScoreFull[0],width=0.4,color='k',align='center')
    plt.bar(np.arange(4,10,2),kplsScoresNys[0,:][::-1],width=0.5,color='c',align='center',label='Nystroem method')
    plt.bar(np.arange(4,10,2)+0.5,kplsScoresRBFS[0,:][::-1],width=0.5,color='y',align='center',label='RBF sampler')
    plt.xticks([2,4.25,6.25,8.25],['Full kernel','1573','1073','573'])
    plt.xlabel('number of kernel components')
    plt.ylabel('accuracies')
    plt.title('Accuracies for several compacted kernel matrices')
    plt.legend(loc='lower right')
    plt.show()
