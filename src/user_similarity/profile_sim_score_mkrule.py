import pandas as pd
import numpy as np

def profile_similairty(host, user, categ_cols, numeric_cols):
    '''
    host: a Dataframe of hosts profiles
    user: a Dataframe of users profiles
    categ_cols: a list of categorical variables
    numeric_cols: a list of numerical variables
    profile: a DataFrame with all active users profile details in active_user table

    return a profile similarity score matrix with n*m dim where n is the number of hosts in
    host Dataframe and m is the number of users in user Dataframe
    Note: the sim score will be in range(0,1), higher scores means higher similarity level.
    '''

    host.reset_index(drop =True, inplace=True) # incase dataframe labels are messed up by rearrange
    user.reset_index(drop =True, inplace=True)
    # print type(host) # make sure its DataFrame
    features = list(set(host.columns.tolist()))#-set(missing_cols)) # not include missing_cols
    # print len(features) # include user_id, len needs to minus 1 cuz the user_id

    #find max difference for each feature between hosts and users
    mx_features=list()
    for f in features:
        mx_features.append(max((max(host[f])-min(user[f])),abs(max(user[f])-min(host[f]))))
    # print mx_features[1:] # exclude the user_id col

    #make zero profile_sim_matrix
    num_users = user.shape[0]
    num_hosts = host.shape[0]
    profile_sim_score_matrix = np.zeros((num_hosts, num_users))
    # print profile_sim_score_matrix.shape
    profile_sim_score_matrix

    for x in range(len(host)):
    #     print host_f.loc[x]
        for y in range(len(user)):
    #         print user_f.loc[y]
            score = 0
            for i in range(1,len(features)):
                # print features[i]
                # print host.loc[x][i],user.loc[y][i]

                if features[i] in categ_cols:
                    if host.loc[x][i] == user.loc[y][i]:
                        score += 1
                    else:
                        score += 0
                    # print score

                if features[i] in numeric_cols:
    #                 print mx_features[i],host.loc[x][i],user.loc[y][i]
                    score += float((mx_features[i] - abs(host.loc[x][i] - user.loc[y][i])))/mx_features[i]
                    # print score
                else:
                    continue

                profile_sim_score = float(score)/(len(features)-1)
                # print profile_sim_score

            profile_sim_score_matrix[x, y] = profile_sim_score
    return profile_sim_score_matrix


if __name__ == "__main__":
    item_data_f = pd.read_csv('item_data_f_rm.csv')
    missing_col = [col for col in item_data_f.columns if 'missing' in col]
    item_data_f = item_data_f.drop(missing_col, axis=1) #accume no correlations on missing cols

    id_cols = ['host_id']
    numeric_cols = ['login_count',
    'photos_count',
    'answers_count',
    'basic_percent',
    'messages_count',
    'replies_count',
    'favored_by_count',
    'visited_by_count',
    'admired_by_count',
    'invitations_count',
    'profile_progress',
    'lat',
    'lng',
    'approved_photos_count',
    'real_photos_count',
    'winked_by_count',
    'reward_points',
    'age',
    'account_length']
    categ_cols = set(item_data.columns.tolist())-set(numeric_cols)

    sim_f = profile_similairty(item_data_f,item_data_f,categ_cols,numeric_cols)
