import csv
import torch
import numpy as np
import getopt, sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import plot_roc_curve , roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, average_precision_score, recall_score

import pandas as pd
import pickle
from numpy import sqrt
from numpy import argmax
from matplotlib import pyplot

################################################################    


def randomforest_ROC_fpr_recallcurve(Xdataset, ydataset):

    # Number of random trials
    NUM_TRIALS = 10 
    average_auc = 0.0

    base_estimator = RandomForestClassifier(max_depth=10, min_samples_split=4, min_samples_leaf=1, n_estimators=300)


    # Loop for each trial
    for i in range(NUM_TRIALS):


        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
        #cv = StratifiedKFold(n_splits=5)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        ytestscore = np.array([])
        ytestlable = np.array([])

        plt.style.use ('~/preprotac/figurestyle/presentation.mplstyle')

        fig, ax_roc = plt.subplots(figsize=(5,4), constrained_layout=True)

        for j, (train, test) in enumerate(cv.split(Xdataset, ydataset)):
            base_estimator.fit(Xdataset[train], ydataset[train])

            modelfile = 'randomforest-esm-all-' + str(i) + '_' + str(j) + '.pkl'

            modeloutput = open(modelfile,'wb')
            pickle.dump(base_estimator,modeloutput)
            modeloutput.close()

            modelinput = open(modelfile,'rb')
            classifier = pickle.load(modelinput)
            modelinput.close()

            viz = plot_roc_curve(classifier, Xdataset[test], ydataset[test],
                                 name='{}'.format(j), ax=ax_roc,alpha=0.3, lw=1)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)


            y_scores = classifier.predict_proba(Xdataset[test])[:,1]
            ytestscore = np.append(ytestscore, y_scores)
            ytestlable = np.append(ytestlable, (ydataset[test]))

        ytestlable.flatten()
        ytestscore.flatten()

        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_xlabel('False Positive Rate')

        ax_roc.plot(line_kw={})
        ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        print("fpr: ", mean_fpr)
        print("tpr: ", mean_tpr)
        std_auc = np.std(aucs)
        print(i, ":" , mean_auc, std_auc)
        average_auc = average_auc + mean_auc
        ax_roc.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=4, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color=[0.2,0.2,0.2], alpha=.3,
                     label=r'$\pm$ 1 std. dev.')

        ax_roc.legend(loc="lower right", fontsize=10)

        handles, labels = ax_roc.get_legend_handles_labels()

        ax_roc.legend(handles[-3:], labels[-3:])


        plt.show()

        figure_file = 'randomforest-esm-all-roc' + str(i) + '.jpg' 

        plt.savefig(figure_file, dpi=600)



        fig, ax_recall = plt.subplots(figsize=(5,4), constrained_layout=True)

        precision, recall, thresholds = precision_recall_curve(ytestlable, ytestscore, pos_label=1)
        average_precision = average_precision_score(ytestlable, ytestscore)
        pr_display = PrecisionRecallDisplay(precision=precision,recall=recall,average_precision=average_precision)
        pr_display.plot(ax=ax_recall)

        ax_recall.plot()
        ax_recall.set_ylabel('Precision')
        ax_recall.set_xlabel('Recall')

        plt.show()

        figure_file = 'randomforest-esm-all-recall' + str(i) + '.jpg'

        plt.savefig(figure_file, dpi=600)

        fig, ax_fpr = plt.subplots(figsize=(5,4), constrained_layout=True)

        fpr, tpr, thresholds = roc_curve(ytestlable, ytestscore, pos_label=1)

        gmeans = sqrt(tpr*(1-fpr))
        ix = argmax(gmeans)

        ax_fpr.plot(thresholds[1:], fpr[1:], marker='.')
        ax_fpr.set_ylabel('False Positive Rate')
        ax_fpr.set_xlabel('Thresholds')
        plt.show()

        figure_file = 'randomforest-esm-all-fpr' + str(i) + '.jpg'
        plt.savefig(figure_file, dpi=600)
 
    average_auc = average_auc / NUM_TRIALS   
    print("average auc: ", average_auc)

################################################################    
