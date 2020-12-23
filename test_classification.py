import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
import random

from tl_algs import trbag, voter

RAND_SEED = 2016 
random.seed(RAND_SEED) # change this to see new random data!


# randomly generate some data
X, domain_index = make_blobs(n_samples=15, centers=3, n_features=2, cluster_std=5)

# randomly assigning domain and label
all_instances = pd.DataFrame({"x_coord" : [x[0] for x in X],
              "y_coord" : [x[1] for x in X],
              "domain_index" : domain_index,
              "label" : [random.choice([True,False]) for _ in X]},
             columns = ['x_coord','y_coord','domain_index', 'label']
            )


#arbitrarily set domain index 0 as target
test_set_domain = 0
# we are going to set the first three instances as test data
# note that this means that some of the target domain has training instances!
test_set = all_instances[all_instances.domain_index == test_set_domain].sample(3, random_state=RAND_SEED)
test_set_X = test_set.loc[:, ["x_coord", "y_coord"]].reset_index(drop=True)
test_set_y = test_set.loc[:, ["label"]].reset_index(drop=True)

# gather all non-test indexes 
train_pool = all_instances.iloc[all_instances.index.difference(test_set.index), ] 
train_pool_X = train_pool.loc[:, ["x_coord", "y_coord"]].reset_index(drop=True)
train_pool_y = train_pool["label"].reset_index(drop=True)
train_pool_domain = train_pool.domain_index


# trbagg
model = trbag.TrBag(test_set_X=test_set_X, 
            test_set_domain=test_set_domain, 
            train_pool_X=train_pool_X, 
            train_pool_y=train_pool_y, 
            train_pool_domain=train_pool_domain, 
            sample_size=test_set_y.shape[0],
            Base_Classifier=RandomForestClassifier,
            filter_func=trbag.mvv_filter,
            validate_proportion=0.5,
            vote_func=voter.mean_confidence_vote,
            rand_seed=RAND_SEED
        )

print(model.train_filter_test())