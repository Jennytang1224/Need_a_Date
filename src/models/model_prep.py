import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def cum_rating(rating_data,item_data):
    '''for item in item_data, calculate the cummulative rating for each item
    to measure a person's attactiveness
    '''
    host = rating[rating['host_id'].isin(item_data['host_id'])]
    cum_sum = pd.DataFrame(host.groupby('host_id')['rating'].sum())
    cum_sum['avg_rating'] = cum_sum['rating']/cum_sum.groupby('host_id')['rating'].count()
    cum_sum = cum_sum.reset_index()
    label = cum_sum['avg_rating']
    return label #cummulated rating for each host

def train_test_split(item_data,label):

    X= item_data.drop('host_id',axis=1) #item_data without id col
    y= label
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test

def gbr(X_train, y_train):
    ''' to find feature importance that affect female rates male and male rates female
    '''
    gbr = GradientBoostingRegressor(min_samples_leaf=3, random_state=0)
    gb.fit(X_train, y_train)
    print 'RMSE: ', np.sqrt(mean_squared_error(y_test, gbr.predict(X_test)))
    feature_importances = gbr.feature_importances_
    feature_scores = pd.DataFrame({'weight' : feature_importances},index = X_train.columns)
    feature_scores = feature_scores.sort_values(by='weight')
    feature_scores

    top_20_features_m = feature_scores.tail(20)
    top_20_features_m.plot(kind='barh', color='green',figsize =(12,10))

def nmf(profile_sim_score):
    '''to find latent features in similarity score matrix '''
    for i in range(1,20): #find i with lowest reconstruction score:
        model= NMF(n_components=i, init='random', random_state=0,alpha=.01)
        W = model.fit_transform(sim)
        print 'reconstruction error:', model.reconstruction_err_

        #add id column to array W, and make W to a DataFrame
        W_to_df = pd.DataFrame(W)
        W_to_df.index = item_data_f['host_id']
        W_to_df.reset_index(inplace=True)
        return W_to_df

def fit_pca():
    sum_rating = pd.DataFrame(rating_f[rating_f['host_id'].isin(item_data_f['host_id'])].groupby('host_id')['rating'].sum()/rating_f['host_id'].value_counts())
    # print max(sum_rating.values),min(sum_rating.values)

    sum_rating.reset_index(inplace=True)
    sum_rating.columns=['host_id','score']
    sum_rating['score'] = sum_rating['score'].round(0)
    # print len(sum_rating['score'])
    sum_rating['score'].value_counts()

    target = sum_rating['score'].values
    df = item_data_f.drop('host_id',axis=1)
    # first 2 PC
    scaler = StandardScaler().fit(df)
    scaled_data=scaler.transform(df)
    pca=PCA(n_components=2)
    pca.fit(scaled_data)
    x_pca=pca.transform(scaled_data)
    # print scaled_data.shape, x_pca.shape

def plot_pca():
    plt.figure(figsize=(12,8))
    plt.scatter(x_pca[:,0],x_pca[:,1],c=target,cmap='plasma')
    #c=cancer['target']: color the points by target categories
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')


def kmeans():
    kmeans = KMeans(n_clusters=4).fit(temp_item_data)
    temp_item_data['kmeans_label'] = kmeans.labels_
    color_map = { 0:'green',1: 'red', 2: 'yellow', 3: 'blue', 4: 'purple'}

    plt.figure(figsize=(8,6))
    plt.scatter(temp_item_data['age'],temp_item_data['rating'], c=temp_item_data['kmeans_label'].map(color_map))
    plt.xlabel('age')
    plt.ylabel('rating')
    plt.title('age vs. rating')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(temp_item_data[''],temp_item_data['rating'], c=temp_item_data['kmeans_label'].map(color_map))
    plt.xlabel('age')
    plt.ylabel('rating')
    plt.title('age vs. rating')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(temp_item_data['photos_count'],temp_item_data['rating'], c=temp_item_data['kmeans_label'].map(color_map))
    plt.xlabel('photos_count')
    plt.ylabel('rating')
    plt.title('photos_count vs. rating')
    plt.show()
