import csv
import torch
import numpy as np
import getopt, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import plot_roc_curve, roc_curve
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, average_precision_score
import pandas as pd
import pickle
from numpy import sqrt
from numpy import argmax
from matplotlib import pyplot

# ###############################################################    
def predict_ROC_fpr(Xdataset, ydataset, modelpath, modelfilename):


    modelfile = modelpath + modelfilename
    modelinput = open(modelfile,'rb')
    classifier = pickle.load(modelinput)
    modelinput.close()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, [ax_roc, ax_fpr] = plt.subplots(1, 2, figsize=(11, 5))


    viz = plot_roc_curve(classifier, Xdataset, ydataset,
                         alpha=0.3, lw=1, ax=ax_roc)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    y_scores = classifier.predict_proba(Xdataset)[:,1]
    print(y_scores)
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_xlabel('False Positive Rate')


    fpr, tpr, thresholds = roc_curve(ydataset, y_scores, pos_label=1)

    gmeans = sqrt(tpr*(1-fpr))
    ix = argmax(gmeans)
    print("ix: ", ix)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    ax_fpr.plot(thresholds[1:], fpr[1:], marker='.')
    ax_fpr.set_ylabel('False Positive Rate')
    ax_fpr.set_xlabel('Thresholds')
    ax_fpr.set_title("Test set fpr thresholds curve")


    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print(mean_auc, std_auc)
    ax_roc.plot(mean_fpr, mean_tpr, color='b',
            label=r'ROC (AUC = %0.2f)' % (mean_auc),
            lw=2, alpha=.8)

    #std_tpr = np.std(tprs, axis=0)
    #tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    #tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color=[0.2,0.2,0.2], alpha=.2,
    #             label=r'$\pm$ 1 std. dev.')

    ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Test set ROC curve")
    ax_roc.legend(loc="lower right")
    plt.show()

    figure_roc_file = 'predict-test-scop-esm-roc.jpg' 

    plt.savefig(figure_roc_file)


################################################################    
def predict_ROC_fpr_recall(Xdataset, ydataset, modelpath, modelfilename):


    modelfile = modelpath + modelfilename
    modelinput = open(modelfile,'rb')
    classifier = pickle.load(modelinput)
    modelinput.close()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.style.use ('/home/lixie/work/protac/training/CRBN/newversion/esm/444-traingset/featuredata/presentation.mplstyle')

    fig, ax_roc = plt.subplots(figsize=(5,4), constrained_layout=True)

    y_scores = classifier.predict_proba(Xdataset)[:,1]

    fpr, tpr, thresholds = roc_curve(ydataset, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_display.plot(ax=ax_roc, lw=4, alpha=.8)
    ax_roc.plot()
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
    ax_roc.legend(loc="lower right")

    handles, labels = ax_roc.get_legend_handles_labels()

    ax_roc.legend(handles[-3:], labels[-3:])

    plt.show()

    figure_file = 'gradientboost-esm-roc-test-scop.jpg'

    plt.savefig(figure_file, dpi=600)

    fig, ax_recall = plt.subplots(figsize=(5,4), constrained_layout=True)

    precision, recall, thresholds = precision_recall_curve(ydataset, y_scores, pos_label=1)
    average_precision = average_precision_score(ydataset, y_scores)
    pr_display = PrecisionRecallDisplay(precision=precision,recall=recall,average_precision=average_precision)
    pr_display.plot(ax=ax_recall)

    ax_recall.plot()
    ax_recall.set_ylabel('Precision')
    ax_recall.set_xlabel('Recall')

    plt.show()

    figure_file = 'gradientboost-esm-recall-test-scop.jpg'

    plt.savefig(figure_file, dpi=600)

    fig, ax_fpr = plt.subplots(figsize=(5,4), constrained_layout=True)

    fpr, tpr, thresholds = roc_curve(ydataset, y_scores, pos_label=1)

    ax_fpr.plot(thresholds[1:], fpr[1:], marker='.')
    ax_fpr.set_ylabel('False Positive Rate')
    ax_fpr.set_xlabel('Thresholds')
    plt.show()

    figure_file = 'gradientboost-esm-fpr-test-scop.jpg'

    plt.savefig(figure_file, dpi=600)

    print("roc_auc: ", roc_auc, "precision: ", average_precision)

################################################################

if  __name__ == "__main__":

    print("get ROC score\n")

