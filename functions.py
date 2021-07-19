import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate


#%% Grab dataset
def get_dr_dataset(filename):
    data = pd.read_csv(filename)
    
    feature = data.iloc[:,:-1]
    target = data.iloc[:,-1]
    
    dataset = data.values
    X = dataset[:,:-1]
    y = dataset[:,-1]
    
    return feature, target, X, y

#%% Normalize
def normalize(data):
    num_array = data.values
    scaler = RobustScaler()
    feature_norm = scaler.fit_transform(num_array)
    feature_norm = pd.DataFrame(feature_norm)
    return feature_norm

#%%  Feature Selection and Evaluation 

# feature selection using Mutual Information
def select_features_mutual(X_train, y_train, X_test, k='all'):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_classif, k=k)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

# evaluate a given model using cross-validation
def evaluate_model(X, y, model, splits=10):
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

#%% Train & Test Machine Learning Model
# train model using knn with cross validation
def train_model_knn(X_train, y_train, n_range=50):
    res_knn = []
    for n in range(1,n_range+1):
       knn = KNeighborsClassifier(n_neighbors=n)
       # Train with cross validation
       scores = evaluate_model(X_train, y_train, model=knn, splits=5)
       res_knn.append(np.mean(scores)*100)
       # summarize the results
       print('>%d %.3f (%.3f)' % (n, np.mean(scores)*100, np.std(scores)))
    print("Skor validasi tertinggi: ", max(res_knn))
    return res_knn , scores

# train model using decision tree with cross validation
def train_model_dt(X_train, y_train, quality, depths=50):
    if quality==1:
        criterion= 'gini'
    else:
        criterion='entropy'
        
    res_dt = []
    for depth in range(1,depths+1):
        dt = DecisionTreeClassifier(random_state=1, criterion=criterion, max_depth=depth)
        # Train with cross validation
        scores = evaluate_model(X_train, y_train, model=dt, splits=5)
        res_dt.append(np.mean(scores)*100)
        # summarize the results
        print('>%d %.3f (%.3f)' % (depth, np.mean(scores)*100,np.std(scores)))
    print("Skor validasi tertinggi: ", max(res_dt))
    return res_dt, scores

# train model using random forest with cross validation
def train_model_rf(X_train, y_train, depths=50):
        
    res_rf = []
    for depth in range(1,depths+1):
       rf = RandomForestClassifier(random_state=1, max_depth=depth)
       # Train with cross validation
       scores = evaluate_model(X_train, y_train, model=rf, splits=5)
       res_rf.append(np.mean(scores)*100)
       # summarize the results
       print('>%d %.3f (%.3f)' % (depth, np.mean(scores)*100,np.std(scores)))
    print("Skor validasi tertinggi: ", max(res_rf))
    return res_rf, scores

# train model using Naive Bayes with cross validation
def train_model_nb(X_train, y_train):
    nb = GaussianNB()
    # Train with cross validation
    scores = evaluate_model(X_train, y_train, model=nb, splits=5)
    # summarize the results
    print('Skor validasi %.3f (%.3f)' % (np.mean(scores)*100,np.std(scores)))
    return scores

# train model using logistic regression with cross validation
def train_model_lr(X_train, y_train):
    res_lr = []
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    for solve in solvers:
        if solve == 'newton-cg' and solve == 'lbfgs':
            lr = LogisticRegression(solver=solve, random_state=8)
        else:
            lr = LogisticRegression(solver=solve)
        # Train with cross validation
        scores = evaluate_model(X_train, y_train, model=lr, splits=5)
        res_lr.append(np.mean(scores)*100)
        # summarize the results
        print('>%s %.3f (%.3f)' % (solve, np.mean(scores)*100, np.std(scores)))
    print("Skor validasi tertinggi: ", max(res_lr))
    return res_lr, scores

# train model using logistic regression with cross validation
def train_model_svc(X_train, y_train, deg=2, coef=3):
    res_svc= []
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    for k in kernels:
        if k== 'sigmoid':
            for n in range (1, coef+1):
                svc = SVC(gamma='auto', kernel='sigmoid', coef0=n, random_state=2)
                scores = evaluate_model(X_train, y_train, model=svc, splits=5)
                res_svc.append(np.mean(scores)*100)
                print('>sigmoid coefficient %i %.3f (%.3f)' % (n, np.mean(scores)*100, np.std(scores)))
        elif k == 'poly':        
            for i in range(1,deg+1):
                for j in range (1,coef+1):
                    svc = SVC(gamma='auto', kernel='poly', degree=i, coef0=j, random_state=2)
                    scores = evaluate_model(X_train, y_train, model=svc, splits=5)
                    res_svc.append(np.mean(scores)*100)
                    print('>poly degree %i coefficient %i  %.3f (%.3f)' % (i, j, np.mean(scores)*100,np.std(scores)))
        else:
            svc = SVC(gamma='auto', kernel=k, random_state=2)
            scores = evaluate_model(X_train, y_train, model=svc, splits=5)
            res_svc.append(np.mean(scores)*100)
            print('>%s %.3f (%.3f)' % (k, np.mean(scores)*100,np.std(scores)))
    print("Skor validasi tertinggi: ", max(res_svc))
    return res_svc, scores

# train model using ANN
def build_ann():
    visible = Input(shape=(19,))
    hidden1 = Dense(18, kernel_initializer='lecun_normal', activation='selu')(visible)
    hidden2 = Dense(7, kernel_initializer='lecun_normal', activation='selu')(hidden1)
    concate = Concatenate()([visible, hidden2])
    output = Dense(1, activation='sigmoid')(concate)
    model = Model(inputs=visible, outputs=output)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model        
    

def test_model(model, X_train, y_train, X_test, y_test):
    # evaluate model prediction
    model = model.fit(X_train, y_train)
    yhat = model.predict(X_test)
        
    model_acc = accuracy_score(y_test, yhat)
    model_rec = recall_score(y_test, yhat)
    model_prec = precision_score(y_test, yhat)
    model_f1 = f1_score(y_test, yhat)
    model_cm = confusion_matrix(y_test, yhat)
    return model_acc, model_rec, model_prec, model_f1, model_cm

def test_ann_model(model, X_train, y_train, X_test, y_test):
    # evaluate model prediction
    yhat = model.predict(X_test)
    yhat = (yhat > 0.5)
        
    model_acc = accuracy_score(y_test, yhat)
    model_rec = recall_score(y_test, yhat)
    model_prec = precision_score(y_test, yhat)
    model_f1 = f1_score(y_test, yhat)
    model_cm = confusion_matrix(y_test, yhat)
    return model_acc, model_rec, model_prec, model_f1, model_cm

#%% RFE
def get_rfe_model(algorithm):
    models= dict()

    for i in range(2, 20):
        rfe = RFE(estimator=algorithm, n_features_to_select=i)
        model = DecisionTreeClassifier()
        models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
    return models        