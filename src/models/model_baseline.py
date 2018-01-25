from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
import operator

def distance(host,user):
    '''
    input:
    user: a numpy array of a list of user's coordinates[lat,lng]
    host: a numpy array of lists of hosts' coordinates[lat,lng]

    return:
    indices of hosts sorted by distance from the user coordinates to all hosts coordinates
    '''
    l = cdist(user,host).argsort()
    return reduce(operator.add, l)

def result(user_id,user_data,host_data,n):
    '''
    input:
    user_data: user profile
    host_data: host profile
    k: choose top k closest distance hosts and return their indices

    return:
    for each user, find the top k closest hosts indices based on their lat and lng
    '''
    user = np.array(user_data[user_data['user_id']==user_id][['lat','lng']])
    host_loc = []
    for index_h, row_h in host_data[['lat','lng']].iterrows():
        host_loc.append(row_h.tolist())
    host = np.asarray(host_loc)
#     print host
    index= distance(host,user)[:n]
    closest_id = []
    for i in index:
        closest_id.append((host_data['host_id'])[i])
    return closest_id

def popularity_model(user_id,closest_hosts_id,host_pop_scores,k):
    '''
    input:
    user: the user index
    hosts: a list of hosts indices available to the user
    host_rating_scores : a Dataframe contains hosts cummulative rating scores from all users

    return:
    for each user, the top n popular hosts indices
    among the k hosts who geographically closest to the user
    '''
    rows_in_pop = f_pop_scores[f_pop_scores['host_id'].isin(closest_hosts)]['pop_score'].sort_values(ascending=False)[:k]
#     print rows_in_pop
    rec_index=rows_in_pop.index.tolist()
    rec=[]
    for index in rec_index:
        rec.append(f_pop_scores['host_id'].iloc[index])
    return rec

if __name__=='__main__':

    user_id = 2878
    host_data_f = pd.read_csv('/Users/jennytang/Desktop/2rb/data/saved/rating/project/item_f.csv')
    user_data_m = pd.read_csv('/Users/jennytang/Desktop/2rb/data/saved/rating/project/exist/user_m_exist.csv')

    f_pop_scores = pd.read_csv('/Users/jennytang/Desktop/2rb/data/saved/rating/project/item_f_pop_score.csv')
    m_pop_scores =pd.read_csv('/Users/jennytang/Desktop/2rb/data/saved/rating/project/item_m_pop_score.csv')
    n=10
    k=5
    closest_hosts = result(user_id,user_data_m,host_data_f,n)
#     print closest_hosts
    print popularity_model(user_id,closest_hosts,f_pop_scores,k)
