import sys
from scipy.sparse import csr_matrix
from scipy import sparse
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from sklearn import preprocessing
import pprint
from pymongo import MongoClient
import pandas as pd
import numpy as np
pd.set_option('chained_assignment', None)
client = MongoClient('localhost', 27017)
reader = Reader(rating_scale=(1, 5))
db = client['database_name']

# Read in ratings

item_ratings_inp = pd.read_csv(sys.argv[1], header=None)
items_tmp = item_ratings_inp.values
item_ratings = {items_tmp[i][0]: items_tmp[i][1]
                for i in range(len(items_tmp))}


# Create dictionary 'coll' of departments per product
asins = pd.read_csv('asins.csv', index_col='asin', compression='gzip')
asins = asins.loc[~asins.index.duplicated(keep='first')]
colls = {}
for i in item_ratings.keys():
    coll = asins.loc[i]['coll']
    if coll in colls.keys():
        colls[coll].append(i)
    else:
        colls[coll] = [i]

# Train model per department
svd_reccs = {}
for coll in colls.keys():
    coll_ = coll.replace('meta_', '') + '_5'
    collection = db[coll_]
    collection_m = db[coll]

    items = colls[coll]

    df_m = pd.DataFrame(collection.find())
    df_m = df_m[['reviewerID', 'asin', 'overall']]
    user_rats = pd.DataFrame({'reviewerID':
                              ['user_' for i in range(len(items))],
                              'asin': [i for i in items],
                              'overall': [item_ratings[i] for i in items]})
    df_m = pd.concat([df_m, user_rats])
    data = Dataset.load_from_df(df_m, reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    items = df_m['asin'].unique()
    reviewed = df_m[df_m['reviewerID'] == 'user_']['asin'].unique()
    no_rev = set(items) - set(reviewed)
    results = []
    for i in list(no_rev):
        uid = 'user_'
        iid = i
        results.append(algo.predict(uid, iid).est)
    preds = pd.DataFrame({'asin': list(no_rev), 'est': results})
    preds = preds.sort_values(by='est', ascending=False).set_index('asin')
    df = pd.DataFrame(collection_m.find())
    preds = preds.join(df[['asin', 'title', 'main_cat']
                          ].set_index('asin'), how='left')
    svd_reccs[coll.replace('meta_', '')] = preds
    print('Finished {} recommendations...'.format(coll.replace('meta_', '')))

# print recommendations
for i in svd_reccs.keys():
    print(f'Recommendations in {i}')
    print(svd_reccs[i]['title'].head(10))
    print()
