import pandas as pd
import numpy as np

def make_flags(visitation,bookmark,favorite,message):
    '''
    make flag for each of the action dafaframe
    '''
    visitation['flag'] = 'v'
    v = visitation[['user_id','host_id','flag']]

    bookmark['flag'] = 'b'
    b = bookmark[['user_id','host_id','flag']]

    favorite['flag'] = 'f'
    f = favorite[['user_id','host_id','flag']]

    message['flag'] = 'm'
    m = message[['user_id','host_id','flag']]

def merge_actions(v,b,f,m):
    '''
    merge all four actions based on user_id and host_id
    '''

    df_temp1 = pd.merge(v, b, how='outer', on=['user_id', 'host_id'])
    df_temp1.rename(columns = {'flag_x':'flag_v', 'flag_y':'flag_b'}, inplace = True)

    df_temp2 = pd.merge(df_temp1, f, how='outer', on=['user_id', 'host_id'])
    df_temp2.rename(columns = {'flag':'flag_f'}, inplace = True)

    df = pd.merge(df_temp2, m, how='outer', on=['user_id', 'host_id'])
    df.rename(columns = {'flag':'flag_m'}, inplace = True)

    print 'number of unique active users have taken actions:', len(df['user_id'].unique())
    print 'number of actions:', len(df)

def convert_actions(row):
    '''
    convert behaviors to ratings:
    message = 4
    favorite = 3
    bookmark = 2
    view = 1

    if message shows in the same row, message(4) will overwirte all other lower rating behaviors(1,2,3)
    if favorite shows in the same row, favorite(3) will overwirte all other lower rating behaviors(1,2)
    if bookmark shows in the same row, bookmark(2) will overwirte all other lower rating behaviors(1)
    '''
    if row['flag_m'] == 'm':
        return '4'
    if row['flag_f'] == 'f':
        return '3'
    if row['flag_b'] == 'b':
        return '2'
    return '1'

def add_ratings(df):
    '''
    apply convert function to df and add rating column to df
    '''
    df['rating'] = df.apply(convert_actions, axis=1)

def save_file(df,path):
    '''
    save file to the path
    '''
    df.to_csv(path,index=False)

def convert_rating_matrix(rating_table):
    '''
    convert rating table to rating matrix
    '''
    rating_matrix = rating_table.pivot(index='user_id', columns='host_id',
    values='rating')
