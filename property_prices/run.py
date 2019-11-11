#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,make_scorer
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import minmax_scale
from itertools import chain, combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV
from collections import defaultdict
from operator import itemgetter


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

train = train.replace(np.nan, -999)
test = test.replace(np.nan, -999)

vip_ = set([11, 12, 24, 48, 53, 65, 78, 83, 86, 130, 133, 143, 196, 205, 227, 231, 240, 277, 298, 302, 320, 325, 331, 339, 363, 366, 377, 386, 389, 396, 406, 408, 421, 431, 433, 434, 437, 449, 459, 471, 472, 482, 485, 495, 514, 528, 529, 543, 547, 552, 564, 574, 583, 584, 587, 592, 593, 602, 614, 617, 630, 636, 642, 646])
#print(vip)

def add_first_floor(df):
    df['is_first_floor'] = (df.loc[:,'floor'] == 1).astype(int)
    if 'is_first_floor' not in COLUMNS:
        COLUMNS.append('is_first_floor')

def add_first_floor(df):
    df['is_first_floor'] = (df.loc[:,'floor'] == 1).astype(int)
    if 'is_first_floor' not in COLUMNS:
        COLUMNS.append('is_first_floor')

def add_vip_street(df):
    df['vip'] = [1 if i in vip else 0 for i in df.loc[:,'street_id'].values]
    if 'vip' not in COLUMNS:
        COLUMNS.append('vip')


def add_metro_dist(df):
    df['near_metro'] = (df.loc[:,'metro_dist']/3).astype(int)

    if 'near_metro' not in COLUMNS:
        COLUMNS.append('near_metro')

def remove_nal(df):
    df = df.drop(df[df['balcon']<0].index, inplace=True)

COLUMNS = ['street_id', 'build_tech', 'floor', 'area', 'rooms', 'balcon', 'g_lift','metro_dist','n_photos','otns','kw1', 'kw2']
#Сортируем индексы улицы по цене кв метра и переименовываем
train['otn']=train.price/train.area
dictx = defaultdict(list)
for idx ,otn in zip(train.street_id.values, train.otn.values):
    dictx[idx]+=[otn]
for key, it in dictx.items():
    dictx[key] =  sum(it) / len(it)

l = [[key,it] for key,it in dictx.items() ]
l =  sorted(l, key=itemgetter(1))
l = {i[0]:ind for i, ind in zip(l, range(len(l)))}
vip = set([l[i] for i in vip_])
train.street_id = [l[i] for i in train.street_id.values]
#==========================================================
train['otns'] = train.area/train.rooms 
y = train['price'].values

submit=True
k_fold =True
#remove_nal(train)
#add_first_floor(train)
#add_vip_street(train)
#pca = PCA(n_components=3)
#pca.fit(Xp)
#X = Xp
#X = pca.transform(Xp)


X = train[COLUMNS].values
skf = StratifiedKFold(n_splits=5, random_state=123,shuffle=True)

if not submit:
    #mdl = RandomForestRegressor(n_estimators=50, n_jobs=7, random_state=123)
    mdl = GradientBoostingRegressor(n_estimators=200, learning_rate=0.3, random_state=123)
    if not k_fold:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        print(mean_absolute_error(y_pred=y_pred, y_true=y_test))
        #scoress[int(mean_absolute_error(y_pred=y_pred, y_true=y_test))] = i
    else:
        scores = np.empty(0,dtype=np.float64)
        for train_k, test_k in skf.split(X, y):
            mdl.fit(X[train_k], y[train_k])
            y_pred = mdl.predict(X[test_k])
            print(mean_absolute_error(y_pred=y_pred, y_true=y[test_k]))
            scores = np.append(scores, mean_absolute_error(y_pred=y_pred, y_true=y[test_k]))
        print('mean:', scores.mean())
else:
 #   add_first_floor(test)
 #   add_vip_street(test)
    test.street_id = [l[i] for i in test.street_id.values]
    #mdl = RandomForestRegressor(n_estimators=800, n_jobs=7, random_state=123)
    mdl = GradientBoostingRegressor(n_estimators=900, learning_rate=0.35, random_state=123)
    test['otns'] = test.area/test.rooms 
    Xt = test[COLUMNS].values
    mdl.fit(X, y)
    y_pred = mdl.predict(Xt)
    test['price'] = y_pred
    test[['id', 'price']].to_csv('sub.csv', index=False)
"""with open('scores', 'w') as out:
    for key in sorted(scoress):
        out.write("{} {}\n".format(key, scoress[key]))"""
#Grid search
"""gsc = GridSearchCV(
    estimator=RandomForestRegressor(n_estimators = 400,n_jobs=7),
    param_grid={
        'max_depth': (1000,2000)
    },
    cv=5, scoring=make_scorer(mean_absolute_error, greater_is_better=False), verbose=0, n_jobs=7)
grid_result = gsc.fit(X, y)
best_params = grid_result.best_params_
print(best_params)
exit(0)"""
