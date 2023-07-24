import csv
import numpy as np
import getopt, sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import pandas as pd

################################################################    

def ridge_gridsearch_ROC_train(Xdataset, ydataset):

    # Number of random trials
    NUM_TRIALS = 30

    # Set up possible values of parameters to optimize over

    param_grid = {"max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                "min_samples_split": [2, 4, 6, 8, 10],
                "n_estimators": [100, 200, 300, 400],
                "min_samples_leaf": [1, 2, 3]}
    # We will use a Support Vector Classifier with "rbf" kernel
    base_estimator = GradientBoostingClassifier()

    # Arrays to store scores
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)

    # Loop for each trial
    for i in range(NUM_TRIALS):

        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(base_estimator, param_grid, cv=inner_cv, 
                    scoring='roc_auc', refit='AUC')

        clf.fit(Xdataset, ydataset)
        print(clf. best_params_)
        non_nested_scores[i] = clf.best_score_
        print("non nested score: ", i, non_nested_scores[i])

        # Nested CV with parameter optimization
        #nested_score = cross_val_score(clf, X=Xdataset, y=ydataset, cv=outer_cv)
        #nested_scores[i] = nested_score.mean()
        #print("nested score: ", i, nested_scores[i])

    #score_difference = non_nested_scores - nested_scores

    #print("Average difference of {:6f} with std. dev. of {:6f}."
    #      .format(score_difference.mean(), score_difference.std()))

################################################################    

if  __name__ == "__main__":

    print("gridsearch validation\n")

