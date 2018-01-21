import pandas as pd
import numpy as np

def profile_similairty(host, user, categ_cols, numeric_cols,weights):
    features = list(set(host.columns.tolist())-set(id_cols)) # not include missing_cols
#     print features
#     print len(features) # include user_id, len needs to minus 1 cuz the user_id
    #find max difference for each feature between hosts and users
    mx_features=list()
    for f in features:
        mx_features.append(max((max(host[f])-min(user[f])),abs(max(user[f])-min(host[f]))))
#     print mx_features[:] # exclude the user_id col
    #make zero profile_sim_matrix
    num_users = user.shape[0]
    num_hosts = host.shape[0]
    profile_sim_score_matrix = np.zeros((num_hosts, num_users))
#     print profile_sim_score_matrix.shape
#     profile_sim_score_matrix
    for x in range(len(host)):
#         print host.loc[x]
        for y in range((x+1),len(user)):
#             print user.loc[y]
            score = 0
            for i in range(1,len(features)):
#                 print features[i]
#                 print host.loc[x][i],user.loc[y][i]
                if features[i] in categ_cols:
                    if host.loc[x][i] == user.loc[y][i]:
                        score += 1 * weights[features[i]]
                    else:
                        score += 0* weights[features[i]]
#                     print score
                if features[i] in numeric_cols:
    #                 print mx_features[i],host.loc[x][i],user.loc[y][i]
                    score += (float((mx_features[i] - abs(host.loc[x][i] - user.loc[y][i])))/mx_features[i])*weights[features[i]]
#                     print score
                profile_sim_score = (float(score)/(len(features)-1))
                # print (x,y)
            profile_sim_score_matrix[x, y] = profile_sim_score
    return profile_sim_score_matrix

if __name__ == "__main__":
item_data_m = pd.read_csv('item_data_m.csv')
item_data_f = pd.read_csv('item_data_f.csv')

id_cols = ['host_id']
numeric_cols = ['login_count','photos_count','lat','lng','age']
categ_cols = set(item_data_f.columns.tolist())-set(numeric_cols)-set(id_cols)

weights_m ={'body_type_1.0': 0.0061123272507610064, 'income_1.0': 0.017357402966591395,
'occupation_15.0': 0.029966378119452141, 'income_0.0': 0.0017905725848638831,
'has_ok_photo_1': 0.0043253589194339071, 'has_ok_photo_0': 0.0029165269086434444,
'email_confirmed_1': 0.0026520826653350692, 'lng': 0.15277806564441568,
'photos_count': 0.20474889663329485, 'marital_status_1.0': 0.00059697671489071416,
'donot_email_1': 0.0020632730870493009, 'login_count': 0.085265849504894298,
'occupation_5.0': 0.0058818609620686279, 'body_type_2.0': 0.020105947359118071,
'body_type_3.0': 0.017706540159800645, 'occupation_14.0': 0.0020074680104071908,
'lat': 0.18865786525987238, 'education_3.0': 0.020477349127479529,
'education_2.0': 0.0015798157292182869, 'age': 0.22405194166791631,
'education_4.0': 0.0035484406065955897, 'marital_status_2.0': 0.00090688897159934482,
'income_2.0': 0.0045021711462984136}

weights_f ={'age': 0.086742139330591869,'body_type_2.0': 0.014813069840614226,
'body_type_3.0': 0.0046685576609842745,'body_type_4.0': 0.015564167466992121,
'donot_email_0': 0.0069751073754606482,'donot_email_1': 0.0038332563285914454,
'education_1.0': 0.0063647039509595812,'education_3.0': 0.036946943165355045,
'education_4.0': 0.002405515054621412,'education_5.0': 0.0023422817763054031,
'email_confirmed_0': 0.0020762690999507629,'have_children_1.0': 0.0005808602352932314,
'have_children_2.0': 0.0016899576670988542, 'income_0.0': 0.0057562679386176808,
'income_2.0': 0.0020988903073775808,'income_4.0': 0.010082336943046026,
'income_6.0': 0.022735376298423593, 'lat': 0.22982118359926518,
'lng': 0.23891893800040775, 'login_count': 0.15472728244863021,
'marital_status_1.0': 0.014607270366520486,'marital_status_2.0': 0.0028016813923944029,
'occupation_13.0': 0.021203301259872749,'occupation_4.0': 0.0047175013473360295,
'occupation_5.0': 0.016950110382519902, 'occupation_6.0': 0.0077379760664728616,
'photos_count': 0.082839054696296599}

sim_m = profile_similairty(item_data_m,item_data_m,categ_cols,numeric_cols,weights_m)
np.save('sim_m',sim_m)

sim_f = profile_similairty(item_data_f,item_data_f,categ_cols,numeric_cols,weights_f)
np.save('sim_f',sim_f)
