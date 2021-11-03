
import time as t
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgbm
import scikitplot as skplt
import matplotlib.pyplot as plt
from hyperopt import fmin as fm 
from sklearn.model_selection import cross_val_score
from hyperopt import (
    STATUS_OK, 
    Trials,
    tpe)
from yellowbrick.classifier import (
    ClassificationReport, 
    ConfusionMatrix, 
    PrecisionRecallCurve, 
    ROCAUC)

SEED = 55

def aux(df):
    
    '''
    in: DataFrame;
    out: DataFrame auxiliar
    '''
    
    
    df_aux = pd.DataFrame(
        {
        'columns': df.columns,
        'dtype': df.dtypes,
        'missing' : df.isna().sum(),
        'size' : df.shape[0],
        'nunique': df.nunique()
        }
        )
    
    df_aux['percentage'] = round(df_aux['missing'] / df_aux['size'], 3)*100
    
    return df_aux

def discount(s1_calc, s2):
    dsc = 100 - ((s2 * 100)/s1_calc)
    dsc = round(dsc, 0)
    if dsc > 0:
        dsc = 1
    elif dsc == 0:
        dsc = 0
    elif dsc < 0:
        dsc = -1
    return dsc

def plot_bars(df, features, target, n_rows, n_cols, title, figsize):
    
    fig = plt.figure(figsize=figsize)
    for i, feat in enumerate(features):
        ax = fig.add_subplot(n_rows,n_cols,i+1)
        sns.countplot(data=df, x=feat, ax=ax, hue=target)
    
    fig.suptitle(title)
    fig.show()

def hyperopt(param_space, X_train, y_train, X_test, y_test, num_eval):
    
    start = t.time()
    
    def objective_function(params):
        clf = lgbm.LGBMClassifier(**params)
        score = cross_val_score(clf, X_train, y_train, cv=5, scoring='recall').mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fm(
        objective_function, 
        param_space, 
        algo=tpe.suggest, 
        max_evals=num_eval, 
        trials=trials,
        rstate= np.random.RandomState(1),
        verbose=-1
    )
    loss = [x['result']['loss'] for x in trials.trials]
    
    best_param_values = [x for x in best_param.values()]
    
    
    if best_param_values[0] == 0:
        boosting_type = 'gbdt'
    else:
        boosting_type= 'dart'
    
    clf_best = lgbm.LGBMClassifier(
        learning_rate=best_param_values[3],
        num_leaves=int(best_param_values[6]),
        max_depth=int(best_param_values[4]),
        n_estimators=int(best_param_values[5]),
        boosting_type=boosting_type,
        colsample_bytree=best_param_values[1],
        reg_lambda=best_param_values[8],
        reg_alpha =  best_param_values[7],
        feature_fraction = best_param_values[2],
        random_state=SEED
        )
    
    clf_best.fit(X_train, y_train, verbose=-1)
    
    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss)*-1)
    print("Best parameters: ", best_param)
    print("Test Score: ", clf_best.score(X_test, y_test))
    print("Time elapsed: ", t.time() - start)
    print("Parameter combinations evaluated: ", num_eval)
    
    return (best_param, clf_best)

def viz_performance(X_train, X_test, y_train, y_test, clf, classes, figsize=(12, 16), cmap='Greens'):
    
    fig, ax = plt.subplots(3, 2, figsize=figsize)
    
    clf = clf.fit(X_train, y_train)
    y_probas = clf.predict_proba(X_test)
    skplt.metrics.plot_ks_statistic(y_test, y_probas, ax=ax[2,1])
    
    grid = [
        ConfusionMatrix(clf, ax=ax[0, 0], classes=classes, cmap=cmap),
        ClassificationReport(clf, ax=ax[0, 1], classes=classes, cmap=cmap ),
        PrecisionRecallCurve(clf, ax=ax[1, 0]),
        ROCAUC(clf, ax=ax[1, 1], micro=False, macro=False, per_class=True, classes=classes)
    ]
    
    for viz in grid:
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()
        
    plt.tight_layout()
    plt.show()