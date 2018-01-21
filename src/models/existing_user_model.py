import numpy as np
import pandas as pd
import graphlab
import graphlab.aggregate as agg
import matplotlib.pyplot as plt
# %matplotlib inline
# import pylab as p
# p.show()

def sframe(df):
    sf_data = graphlab.SFrame(df)
    return sf_data

def train_test_split(rating):
    training_data, test_data = graphlab.recommender.util.random_split_by_user(rating,
    user_id='user_id',
    item_id='host_id',
    )

    training_data, validation_data = graphlab.recommender.util.random_split_by_user(training_data,
    user_id='user_id',
    item_id='host_id',
    item_test_proportion=0.2)
    return training_data, validation_data, test_data

def ranking_model(training_data,validation_data,item_data):
    '''
    training_data: SFrame
    validation_data: SFrame
    item_data: either items_profile data or W from NMF as latent feature
    '''
    model_rank= graphlab.ranking_factorization_recommender.create(training_data,
    user_id = 'user_id',
    item_id = 'host_id',
    target = 'rating',
    item_data = W) #W comes from NMF
    rmse_results_rank = model_rank.evaluate(validation_data)

    model_rank_m.save('model_rank_m')

        #f:
    Precision and recall summary statistics by cutoff
    +--------+-------------------+------------------+
    | cutoff |   mean_precision  |   mean_recall    |
    +--------+-------------------+------------------+
    |   1    |        0.0        |       0.0        |
    |   2    |        0.0        |       0.0        |
    |   3    |        0.0        |       0.0        |
    |   4    |        0.0        |       0.0        |
    |   5    |        0.0        |       0.0        |
    |   6    |        0.0        |       0.0        |
    |   7    |        0.0        |       0.0        |
    |   8    |        0.0        |       0.0        |
    |   9    |        0.0        |       0.0        |
    |   10   | 0.000126742712294 | 0.00063371356147 |
    +--------+-------------------+------------------+
    [10 rows x 3 columns]

    ('\nOverall RMSE: ', 1.5331676492036377)

        #M:
    recision and recall summary statistics by cutoff
    +--------+-------------------+-------------------+
    | cutoff |   mean_precision  |    mean_recall    |
    +--------+-------------------+-------------------+
    |   1    |        0.0        |        0.0        |
    |   2    |        0.0        |        0.0        |
    |   3    | 0.000397772474145 | 0.000298329355609 |
    |   4    | 0.000298329355609 | 0.000298329355609 |
    |   5    | 0.000238663484487 | 0.000298329355609 |
    |   6    | 0.000198886237072 | 0.000298329355609 |
    |   7    | 0.000170473917491 | 0.000298329355609 |
    |   8    | 0.000149164677804 | 0.000298329355609 |
    |   9    |  0.00026518164943 | 0.000894988066826 |
    |   10   | 0.000238663484487 | 0.000894988066826 |
    +--------+-------------------+-------------------+
    [10 rows x 3 columns]

    ('\nOverall RMSE: ', 1.9758193654729248)


def compare_models(list_of_models,metric):
    '''
    list_of_models: model names in a list
    metric: ‘rmse’ or ‘precision_recall’
    '''
    agg_list = [agg.AVG('precision'),agg.STD('precision'),agg.AVG('recall'),agg.STD('recall')]
    # apply above functions to each group(group the results by cutoff k which is the number of top items to look for
    print rmse_results['precision_recall_by_user'].groupby('cutoff',agg_list)

    comparisonstruct = graphlab.recommender.util.compare_models(test_data,model_names=list_of_models,metric=metric)
    return graphlab.show_comparison(comparisonstruct,list_of_models)

def make_rec(model,id):
    '''
    use model to recommend matchese to the user has the id number
    '''
    result = model.recommend([id])
    return results.print_rows(num_rows=20, num_columns=4)

if __name__=='__main__':
    # female
    rating_m = pd.read_csv('rating_m_exist.csv') # female rate male
    item_m = pd.read_csv('item_m.csv') # male hosts profile data
    W_m = pd.read_csv('W_m_df.csv')
    rating_f = pd.read_csv('rating_f_exist.csv') # female rate male
    item_f = pd.read_csv('item_f.csv') # male hosts profile data
    W_f= pd.read_csv('W_f_df.csv')
    # user_data_f = pd.read_csv('user_data_f.csv') # female users profile data

    # sim_m = np.load('sim_f.npy') # similarity score between all male hosts

    action_m = graphlab.SFrame(rating_m)
    item_data_m = graphlab.SFrame(item_m)
    W_m= graphlab.SFrame(W_m)
    action_f= graphlab.SFrame(rating_f)
    item_data_f = graphlab.SFrame(item_f)
    W_f= graphlab.SFrame(W_f)

    training_data,validation_data,test_data = train_test_split(action_m)

    rec = models(training_data,validation_data)
