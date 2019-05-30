import numpy as np
import pandas as pd
import pydotplus
from models import common
from os.path import join
from os import remove
from sklearn import tree
from operator import itemgetter
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix


pydotplus.find_graphviz()


def build_model(data_set, data_split, no_interactions, negative, max_snps, cross_validation, output_dir):
    param_grid = {
        "criterion": ["gini", "entropy"],
          #If float then min_samples_split is a percentage and ceil(min_samples_split * n_samples)
            # are the minimum number of samples for each split.
          "min_samples_split": [.01, .015, .02, .025],
          "max_depth": [None, 4, 5], # int or None. None allows a full tree
          #If float then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples)
            #  are the minimum number of samples for each leaf.
          "min_samples_leaf": [.0025, .005, .01,.015], # Berry and Linoff .0025 to .01
          #If float then max_features is a percentage and int(min_max_features* n_features)
            # features are considered at each split. If 'auto' then max_features = sqrt(n_features)
            # If None then max_features = n_features
          "max_features":[.3,.4,.5,] #A Complete Tutorial on Tree Based Modeling from Scratch (in R & Python)is 30 to 40%
      }

    model_eval = {
        'features': save_features
    }

    common.build_model(
        data_set,
        data_split,
        True,
        negative,
        tree.DecisionTreeClassifier(random_state=1),
        cross_validation,
        max_snps,
        output_dir,
        param_grid,
        model_eval
    )


def save_features(model, term_labels, output_dir):
    dot_file = join(output_dir, "dtree.dot")
    with open(dot_file, 'w') as f:
        tree.export_graphviz(model, out_file=f, feature_names=term_labels)

    graph = pydotplus.graphviz.graph_from_dot_file(dot_file)
    graph.write_png(join(output_dir, 'dtree.png'))
    remove(dot_file)


# Chris Strelioff's code: replace this as it is deprecated
#http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html
def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.
    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters
