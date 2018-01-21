from scipy.spatial.distance import cdist
import sklearn.metrics.pairwise
import pandas as pd
import numpy as np
import operator
import itertools

def calculate_distance(host,user):
    '''
    input:
    user: a numpy array of a list of user's coordinates[lat,lng]
    host: a numpy array of lists of hosts' coordinates[lat,lng]

    return:
    the distance from the user coordinates to all hosts coordinates
    '''
    l = cdist(user,host).argsort()
    return reduce(operator.add, l)

def find_closest(user_data,host_data,n):
    '''
    input:
    user_data: user profile
    host_data: host profile
    k: choose top k closest distance hosts and return their indices

    return:
    for each user, find the top k closest hosts based on their lat and lng
    '''
    for index_u, row_u in user_data[['lat','lng']].iterrows():
        lst = row_u.tolist()
        user = np.array([lst])
    #     print user
        host_loc = []
        for index_h, row_h in host_data[['lat','lng']].iterrows():
            host_loc.append(row_h.tolist())
        host = np.asarray(host_loc)
    #     print host
        dist = calculate_distance(host,user)[:n]
        return dist

def cos_sim(X,Y):
    m = sklearn.metrics.pairwise.cosine_similarity(X,Y)
    return (m)

def find_sim_hosts(cold_user_id, user_profile, hosts_profile, k):
    '''
    input:
    cold_user_id: the user id needs recommendations
    cold user_profile: this cold user's profile
    hosts_profile: dataframe of all other same gender user profile
    k: top k most similar hosts to cold user

    output:
    a list of hosts that most similar to the user
    '''
    user_data = user_profile[user_profile['user_id'] == cold_user_id]
    host_data = hosts_profile
    closest_lst = find_closest(user_data,host_data,n) # location filter get closest hosts like cold user
#     print closest_lst

    store = dict()
    for i in closest_lst: # i is index of the host # profile_sim filter among the closest users
        host = np.array(hosts_profile.iloc[i].values.tolist()).reshape(1,-1)
#         print host
        user = user_data.as_matrix()
        sim = cos_sim(user,host) # or use profile_sim_score_mkrule.py
        key = hosts_profile['host_id'][i] # convert index to host_id
        store[key] = sim #dict{index:sim_score}
#     print store

    top_sim_lst = []
    for key, value in sorted(store.iteritems(), key=lambda (key,value): (value,key), reverse=True)[:k]:
#         value = value[0,0]
#         print "%s: %s" % (key, value)
        top_sim_lst.append(key)
    return top_sim_lst

def recommendation(cold_user_id,top_sim_lst,m):
    '''
    top_sim_lst: a list of user ids that most similar profile and nearby location to cold start user
    m: send m recomendations
    '''
    rec_lst =[]
    for id_ in list(top_sim_lst):
#         print id_
        match1 = rating_active_f[rating_active_f['host_id'] ==cold_user_id]['user_id'].unique()
        match2 = rating_active_f[rating_active_f['host_id'] ==id_]['user_id'].unique()

        rec_lst.append(match1)
        rec_lst.append(match2)
#     print rec_lst
    x =list(itertools.chain(*rec_lst))
    rec = np.random.choice(x,m,replace = False) # random sample m recomendations w/o replacement from the rec_list
    return rec

if __name__ == '__main__':
    # example
    rating_active_f = pd.read_csv('rating_f_rm.csv')
    rating_active_m = pd.read_csv('rating_m_rm.csv')
    cold_user_id = 21104
    user_profile = user_m_cold
    hosts_profile = item_m
    n=50 # top closest distance
    k=30 # top profile_sim
    m=10 # num of rec

    top_sim_lst = find_sim_hosts(cold_user_id, user_profile, hosts_profile, k )
#     print top_sim_lst
    print recommendation(cold_user_id,top_sim_lst,m)
