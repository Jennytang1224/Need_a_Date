import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import sklearn.metrics.pairwise
import operator
import graphlab
import itertools

# baseline_model
def calculate_distance(host,user):
    '''
    input:
    user: a numpy array of a list of user's coordinates[lat,lng]
    host: a numpy array of lists of hosts' coordinates[lat,lng]

    return:
    indices of hosts sorted by distance from the user coordinates to all hosts coordinates
    '''
    l = cdist(user,host).argsort()
    return reduce(operator.add, l)

def find_closest(user_id, user_data, host_data,n):
    '''
    return
    for each user, find the top n closest hosts indices based on their lat and lng
    '''
    user = np.array(user_data[user_data['user_id']==user_id][['lat','lng']])

    host_loc = []
    for index_h, row_h in host_data[['lat','lng']].iterrows():
        host_loc.append(row_h.tolist())
    host = np.asarray(host_loc)
#     print host
    index= calculate_distance(host,user)[:n]
    closest_id = []
    for i in index:
        closest_id.append((host_data['host_id'])[i])
    return closest_id

def baseline_model(user_id, user_data, host_data, host_pop_scores, n,k):
    '''
    input:
    user_id: the user_id
    hosts: a list of hosts indices available to the user
    host_rating_scores : a Dataframe contains hosts cummulative rating scores from all users
    user_data: user profile
    host_data: host profile
    k: choose top k closest distance hosts and return their indices

    return:
    for each user, the top n popular hosts indices
    among the k hosts who geographically closest to the user
    '''
    closest_host_ids = find_closest(user_id, user_data, host_data,n)
    rows_in_pop = host_pop_scores[host_pop_scores['host_id'].isin(closest_host_ids)]['pop_score'].sort_values(ascending=False)[:k]
#     print rows_in_pop
    rec_index=rows_in_pop.index.tolist()
    rec=[]
    for index in rec_index:
        rec.append(host_pop_scores['host_id'].iloc[index])
    return rec

# model for existing users
def rank_fact_model(model_rank, id_, n): # model saved from model_exisiting_user
    result_rank = model_rank.recommend([id_],k=n)
    return (result_rank['host_id']).to_numpy()

#model for cold users
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
    return closest_lst

    store = dict()
    for i in closest_lst: # i is index of the host # profile_sim filter among the closest users
        host = np.array(hosts_profile.iloc[i].values.tolist()).reshape(1,-1)
#         print host
        user = user_data.as_matrix()
        sim = cos_sim(user,host)
        key = hosts_profile['host_id'][i] # convert index to host_id
        store[key] = sim #dict{index:sim_score}
#     print store

    top_sim_lst = list()
    for key, value in sorted(store.iteritems(), key=lambda (key,value): (value,key), reverse=True)[:k]:
        top_sim_lst.append(key)
    return top_sim_lst

def cold_recommendation(cold_user_id,user_profile, hosts_profile, k, m,rating):
    '''
    top_sim_lst: a list of user ids that most similar profile and nearby location to cold start user
    m: send m recomendations
    '''
    rec_lst =[]
    top_sim_lst = find_sim_hosts(cold_user_id, user_profile, hosts_profile, k)
    for id_ in top_sim_lst:
        match1 = rating[rating['host_id'] ==cold_user_id]['user_id'].unique()
        match2 = rating[rating['host_id'] ==id_]['user_id'].unique()

        rec_lst.append(match1)
        rec_lst.append(match2)
#     print rec_lst
    x =list(itertools.chain(*rec_lst))
    rec = np.random.choice(x,m,replace = False) # random sample m recomendations w/o replacement from the rec_list
    return rec


# evaluation model
def precision_recall_for_one(id_, rec_lst, rating):
    user_rating = rating[rating['user_id'] == id_]
    print 'number of user ratings:', len(user_rating)
    rated_rec = user_rating[(user_rating['host_id'].isin(rec_lst)) & (user_rating['rating'] !=1)]

    print rated_rec
    num_rated_rec = len(rated_rec)
    print 'number of rated rec among all rec:' , num_rated_rec
    num_total_rec = len(rec_lst)
    print 'number of total rec' , num_total_rec

    #precision:
    if rated_rec.empty == True:
        precision = 0
        recall = 0
    else:
        precision = float(num_rated_rec)/num_total_rec
        #recall:
        sum_rated_rec = rated_rec['rating'].sum()
        recall = float(sum_rated_rec)/num_rated_rec
    return precision, recall

def precision_recall_for_all(samp_users, model, rating, q,n,user_data, host_data, host_pop_scores, model_rank):
    '''
    samp_user: sampled user_table, need to extract user_id
    model: choose 'baseline_model' or 'rank_fact_model' to output recmendations
    rating: rating table need to
    q: top q recomendations
    '''
    user_ids = samp_users['user_id'].tolist()
#     print user_ids
    precision_lst = []
    recall_lst = []
    for id_ in user_ids:
#         print id_
        if model == 'baseline_model':
            rec_lst = baseline_model(id_, user_data, host_data, host_pop_scores, n, q) # n > q
#             print rec_lst
        else:
            if user_data['user_id'].isin([id_]):
                rec_lst = rank_fact_model(model_rank, id_, q)
            else:
                rec_lst = recommendation(user_id,user_data, host_data, n, q,rating)
#         print rec_lst
        p, r = precision_recall_for_one(id_, rec_lst, rating)
#         print p,r
        precision_lst.append(p)
        recall_lst.append(r)
    precision = np.mean(precision_lst)
    recall = np.mean(recall_lst)
    return precision, recall


if __name__=='__main__':

    user_data_m = pd.read_csv('user_m_exist.csv')
    user_data_f = pd.read_csv('user_f_exist.csv')

    host_data_m = pd.read_csv('item_m.csv')
    host_data_f = pd.read_csv('item_f.csv')

    f_pop_scores =pd.read_csv('item_m_pop_score.csv')
    m_pop_scores = pd.read_csv('item_f_pop_score.csv')

    rating_m = pd.read_csv('rating_active_m.csv')
    rating_f = pd.read_csv('rating_active_f.csv')

    model_rank_m = graphlab.load_model('model_rank_m')
    model_rank_f = graphlab.load_model('model_rank_f')

    samp_count_m =pd.read_csv('samp_count_m.csv')
    samp_count_f =pd.read_csv('samp_count_f.csv')


    '''for men: '''

    host_pop_scores = f_pop_scores
    rating = rating_m_fullyear
    n= 400
    q = 200 # cutoff

    model = 'baseline_model'
    print precision_recall_for_all(samp_users_m, model, rating, q,n, user_data_m, host_data_m, host_pop_scores,model_fact_m)

    model = 'rank_fact_model'
    print precision_recall_for_all(samp_users_m, model, rating, q,n, user_data_m, host_data_m, host_pop_scores, model_fact_m)

    '''for women: '''
    user_data = user_data_f
    host_data = host_data_f
    host_pop_scores = m_pop_scores
    rating = rating_f_fullyear
    n = 400
    q= 200

    model = 'baseline_model'
    print precision_recall_for_all(samp_users_f, model, rating, q,n, user_data_f, host_data_f, host_pop_scores, model_fact_f)

    model = 'rank_fact_model'
    print precision_recall_for_all(samp_users_f, model, rating, q,n, user_data_f, host_data_f, host_pop_scores, model_fact_f)
