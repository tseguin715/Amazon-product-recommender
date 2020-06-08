from nltk.stem.porter import PorterStemmer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from surprise.model_selection import cross_validate
from surprise import NMF
from surprise import accuracy
from surprise.model_selection import KFold
from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from surprise import SVD
from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
import pprint
from pymongo import MongoClient
import pandas as pd
import numpy as np
pd.set_option('chained_assignment', None)
client = MongoClient('localhost', 27017)
reader = Reader(rating_scale=(1, 5))
db = client['database_name']


def compute_score(predicted, actual):
    """Compute scores for predicted ratings:

    Score 1: Average actual rating for products predicted in top 5% per user
    Score 2: RMSD for products predicted to to be in top 5% per user
    Score 3: Overall RMSD

    Args:
        predicted: Pandas dataframe with 'reviewerID',
        'asin','overall' columns (predicted ratings)
        actual: Pandas dataframe with
    'reviewerID','asin','overall' columns (actual ratings)

    Returns:
        tuple: (Score1,Score2,Score3)

    """
    df_tmp = pd.merge(predicted, actual, on=['reviewerID', 'asin']).fillna(1.0)
    top_5 = df_tmp.groupby('reviewerID')['overall_x'].transform(
        lambda x: x >= x.quantile(.95))
    val1 = df_tmp[top_5]['overall_y'].mean()
    val2 = (mean_squared_error(
        df_tmp[top_5]['overall_y'], df_tmp[top_5]['overall_x']))**(1 / 2)
    val3 = (mean_squared_error(
        df_tmp['overall_y'], df_tmp['overall_x']))**(1 / 2)
    return val1, val2, val3


def my_tokenizer(doc):
    """Tokenizes document using RegExpTokenizer

    Args:

        doc: string

    Returns:
        list: tokenized words

    """

    tokenizer = RegexpTokenizer(r'\w+')
    article_tokens = tokenizer.tokenize(doc.lower())
    return article_tokens


def get_dept(dept):
    """Returns a dataframe containing ratings information
    (df) and a dataframe with cleaned
    meta data for a department (df_m)


    Args:
        dept: string

    Returns:
        tuple: (df,df_m)

    """

    collection = db[dept + '_5']
    df = pd.DataFrame(collection.find())

    collection_m = db['meta_' + dept]
    df_m = pd.DataFrame(collection_m.find())

    df_m = df_m[~df_m['title'].str.contains('var aPageStart').fillna(True)]
    lst = ['category', 'feature', 'brand', 'description']
    df_m['bow'] = ' '
    for i in lst:
        df_m[i] = df_m[i].fillna(' ').apply(
            lambda x: ' '.join(x) if isinstance(
                x, list) else str(x))
        df_m['bow'] += df_m[i]
    df_m = df_m.sort_values(
        by='asin').drop_duplicates(
        subset='asin',
        keep='last')
    return df, df_m


def train_test_split(df, df_m, main_cat=None, min_revs=10):
    """Split user rating data by time in half to
        train (df_train) and test portions (df_test)


    Args:
        df: Pandas dataframe user ratings
        df_m: Pandas dataframe meta data
        main_cat: main category for department to use (optional)
        min_revs: minimum number of total reviews
    per user between train and test portions

    Returns:
        tuple: (df_train,df_test)

    """

    if main_cat:
        mask = df_m[df_m['main_cat'] == main_cat]['asin']
        df_test = df[df['asin'].isin(mask)]
    else:
        df_test = df.copy()
    df_test['reviewTime'] = df_test['reviewTime'].apply(
        lambda x: int(x.split(', ')[-1]))
    df_test.drop(columns='_id', inplace=True)
    df_test = df_test[['reviewTime', 'reviewerID',
                       'asin', 'overall']].drop_duplicates(keep='last')

    df_test = df_test[df_test['asin'].isin(df_m['asin'].unique())]

    midpoint = int(df_test.shape[0] * 0.5)
    df_train = df_test.sort_values(by='reviewTime').iloc[:midpoint, :]
    df_test = df_test.sort_values(by='reviewTime').iloc[midpoint:, :]

    revs_tot = df_train['reviewerID'].value_counts().reset_index().merge(
        df_test['reviewerID'].value_counts().reset_index(), on='index')
    revs_tot['tot'] = revs_tot['reviewerID_x'] + revs_tot['reviewerID_y']
    reviewers = revs_tot.sort_values(by='tot', ascending=False)
    reviewers = reviewers[reviewers['tot'] > min_revs]
    reviewers = set(list(reviewers['index']))

    shared_reviewers = list(set(df_train['reviewerID'].unique()).intersection(
        set(df_test['reviewerID'].unique()))).copy()
    shared_reviewers = list(set(shared_reviewers).intersection(reviewers))

    df_test = df_test[df_test['reviewerID'].isin(shared_reviewers)]
    df_train = df_train[df_train['reviewerID'].isin(shared_reviewers)]
    return df_test, df_train


# Collect scores per deparment in 'depts_'
depts_ = ['Software']

reg_scores = []
sim_scores = []
svd_scores = []
nmf_scores = []

for dept_ in depts_:
    df, df_m = get_dept(dept_)
    df_test, df_train = train_test_split(df, df_m)

    # Collect SVD scores
    data = Dataset.load_from_df(
        df_train[['reviewerID', 'asin', 'overall']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    results = []
    for i in df_test[['reviewerID', 'asin']].values:
        uid = i[0]
        iid = i[1]
        results.append(algo.predict(uid, iid).est)

    df_preds = df_test[['reviewerID', 'asin']]
    df_preds['overall'] = results

    svd_scores.append(compute_score(
        df_preds, df_test[['reviewerID', 'asin', 'overall']]))
    print('Finished SVD')

    # Collect NMF scores
    data = Dataset.load_from_df(
        df_train[['reviewerID', 'asin', 'overall']], reader)
    trainset = data.build_full_trainset()
    algo = NMF()
    algo.fit(trainset)

    results = []
    for i in df_test[['reviewerID', 'asin']].values:
        uid = i[0]
        iid = i[1]
        results.append(algo.predict(uid, iid).est)

    df_preds = df_test[['reviewerID', 'asin']]
    df_preds['overall'] = results

    nmf_scores.append(compute_score(
        df_preds, df_test[['reviewerID', 'asin', 'overall']]))
    print('Finished NMF')

    # Collect XGBoost scores
    df_reg_preds = pd.DataFrame({'reviewerID': [], 'asin': [], 'overall': []})
    df_reg_tests = pd.DataFrame({'reviewerID': [], 'asin': [], 'overall': []})

    for rev in list(df_test['reviewerID'].value_counts().index):
        a = list(df_train[df_train['reviewerID'] == rev]['asin'].unique())
        b = list(df_test[df_test['reviewerID'] == rev]['asin'].unique())
        c = list(set(a + b))
        df_sim_train = df_m[df_m['asin'].isin(c)]

        df_m2 = df_sim_train
        asins = list(df_m2['asin'])

        vectorizer = TfidfVectorizer(
            tokenizer=my_tokenizer,
            stop_words='english',
            max_features=5000)
        mat = vectorizer.fit_transform(df_m2['bow']).toarray()

        df_reg = pd.DataFrame(index=asins, data=mat)
        df_reg_train = df_reg[df_reg.index.isin(a)]
        df_reg_test = df_reg[df_reg.index.isin(b)]

        tmp = df_train[df_train['asin'].isin(
            a)][['asin', 'overall', 'reviewerID']]
        rat_tmp = tmp[tmp['reviewerID'] == rev][['asin', 'overall']].values
        item_ratings = {rat_tmp[i][0]: rat_tmp[i][1]
                        for i in range(len(rat_tmp))}

        df_reg_train_y = [item_ratings[i] for i in df_reg_train.index]

        model = XGBRegressor()
        model.fit(csc_matrix(df_reg_train), df_reg_train_y)

        tmp = df_test[df_test['asin'].isin(
            b)][['asin', 'overall', 'reviewerID']]
        rat_tmp = tmp[tmp['reviewerID'] == rev][['asin', 'overall']].values
        item_ratings = {rat_tmp[i][0]: rat_tmp[i][1]
                        for i in range(len(rat_tmp))}
        y_test = pd.Series([item_ratings[i] for i in df_reg_test.index])

        preds = model.predict(csc_matrix(df_reg_test))
        reg_test_final = pd.DataFrame(
            {'reviewerID': rev, 'asin': df_reg_test.index, 'overall': y_test})
        reg_preds_final = pd.DataFrame(
            {'reviewerID': rev, 'asin': df_reg_test.index, 'overall': preds})

        df_reg_preds = pd.concat([df_reg_preds, reg_preds_final])
        df_reg_tests = pd.concat([df_reg_tests, reg_test_final])

    reg_scores.append(compute_score(df_reg_preds, df_reg_tests))
    print('Finished XGBoost')

    # collect cosine similarity scores
    df_sim_preds = pd.DataFrame({'reviewerID': [], 'asin': [], 'overall': []})
    df_sim_tests = pd.DataFrame({'reviewerID': [], 'asin': [], 'overall': []})

    tot_ = len(list(df_test['reviewerID'].value_counts().index))
    num_ = 0

    for rev in list(df_test['reviewerID'].value_counts().index):

        a = list(df_train[df_train['reviewerID'] == rev]['asin'].unique())
        b = list(df_test[df_test['reviewerID'] == rev]['asin'].unique())
        c = list(set(a + b))
        df_m2 = df_m[df_m['asin'].isin(c)]

        asins = list(df_m2['asin'])

        vectorizer = TfidfVectorizer(
            tokenizer=my_tokenizer,
            stop_words='english',
            max_features=5000)
        mat = vectorizer.fit_transform(df_m2['bow']).toarray()
        df_m2 = pd.DataFrame(data=mat.T, columns=asins)

        tmp = df_train[df_train['asin'].isin(
            a)][['asin', 'overall', 'reviewerID']]
        rat_tmp = tmp[tmp['reviewerID'] == rev][['asin', 'overall']].values
        item_ratings_inp = {rat_tmp[i][0]: rat_tmp[i][1]
                            for i in range(len(rat_tmp))}

        item_ratings = item_ratings_inp

        df_m2['user_col'] = 0

        items = list(item_ratings.keys())

        rating_sum = 1

        for _ in range(len(items)):
            weight = item_ratings[items[_]]
            if weight >= 4:
                df_m2['user_col'] += weight * df_m2[items[_]]
                rating_sum += weight
            else:
                continue

        df_m2['user_col'] = df_m2['user_col'] / rating_sum

        df_m2 = pd.DataFrame(
            cosine_similarity(
                np.float32(
                    df_m2.values.T)),
            index=df_m2.columns,
            columns=df_m2.columns)
        df_m2 = df_m2['user_col'].sort_values(
            ascending=False).drop_duplicates(
            keep="last").reset_index()

        df_ins = df_m2.set_index('index')
        df_ins = df_ins.join(
            df_m[['asin', 'title']].set_index('asin'), how='left')
        df_sim_pred = pd.DataFrame(
            {'reviewerID': rev, 'asin': df_ins.index,
             'overall': df_ins['user_col']})

        df_sim_pred = df_sim_pred[df_sim_pred['asin'].isin(b)]
        df_sim_test = df_test[df_test['reviewerID']
                              == rev][['reviewerID', 'asin', 'overall']]

        df_sim_preds = pd.concat([df_sim_preds, df_sim_pred])
        df_sim_tests = pd.concat([df_sim_tests, df_sim_test])

    sim_scores.append(compute_score(df_sim_preds, df_sim_tests))

    print(f'Finished {dept_}')
    ####

# create and save dataframe of scores
depts_tmp = depts_
all_scores = pd.DataFrame({'dept': [],
                           'svd_scores': [],
                           'reg_scores': [],
                           'sim_scores': [],
                           'nmf_scores': []})
for i in range(len(depts_tmp)):
    tmp = {
        'dept': depts_tmp[i],
        'svd_scores': svd_scores[i],
        'reg_scores': reg_scores[i],
        'sim_scores': sim_scores[i],
        'nmf_scores': nmf_scores[i]}
    all_scores = all_scores.append(tmp, ignore_index=True)

all_scores.to_csv('all_scores.csv')
