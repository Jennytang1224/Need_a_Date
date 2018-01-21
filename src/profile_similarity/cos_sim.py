import sklearn.metrics.pairwise
import pandas as pd
import numpy as np

def cos_sim(item_data):
    m = sklearn.metrics.pairwise.pairwise_distances(item_data, metric='cosine', n_jobs=1)
    return (1-m)

if __name__=='__main__':
    item_data_m = pd.read_csv('item_m.csv')
    sim_m = cos_sim(item_data_m)
    print sim_m.shape

    np.save('sim_m',sim_m)
