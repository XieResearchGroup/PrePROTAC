import csv
import torch
import numpy as np
import getopt, sys
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
import pandas as pd
import pickle


# ###############################################################    
def randomforest_esm_get_voting(datapath):

    models = list()

    modelfile1 = datapath + 'randomforest-esm-all_0.pkl'
    model1input = open(modelfile1,'rb')
    classifier1 = pickle.load(model1input)
    model1input.close()
    models.append(('rf1',classifier1))

    modelfile2 = datapath + 'randomforest-esm-all_1.pkl'
    model2input = open(modelfile2,'rb')
    classifier2 = pickle.load(model2input)
    model2input.close()
    models.append(('rf2',classifier2))

    modelfile3 = datapath + 'randomforest-esm-all_2.pkl'
    model3input = open(modelfile3,'rb')
    classifier3 = pickle.load(model3input)
    model3input.close()
    models.append(('rf3',classifier3))

    modelfile4 = datapath + 'randomforest-esm-all_3.pkl'
    model4input = open(modelfile4,'rb')
    classifier4 = pickle.load(model4input)
    model4input.close()
    models.append(('rf4',classifier4))

    modelfile5 = datapath + 'randomforest-esm-all_4.pkl'
    model5input = open(modelfile5,'rb')
    classifier5 = pickle.load(model5input)
    model5input.close()
    models.append(('rf5',classifier5))

    modelfile6 = datapath + 'randomforest-esm-all_5.pkl'
    model6input = open(modelfile6,'rb')
    classifier6 = pickle.load(model6input)
    model6input.close()
    models.append(('rf6',classifier6))

    modelfile7 = datapath + 'randomforest-esm-all_6.pkl'
    model7input = open(modelfile7,'rb')
    classifier7 = pickle.load(model7input)
    model7input.close()
    models.append(('rf7',classifier7))

    modelfile8 = datapath + 'randomforest-esm-all_7.pkl'
    model8input = open(modelfile8,'rb')
    classifier8 = pickle.load(model8input)
    model8input.close()
    models.append(('rf8',classifier8))

    modelfile9 = datapath + 'randomforest-esm-all_8.pkl'
    model9input = open(modelfile9,'rb')
    classifier9 = pickle.load(model9input)
    model9input.close()
    models.append(('rf9',classifier9))

    modelfile10 = datapath + 'randomforest-esm-all_9.pkl'
    model10input = open(modelfile10,'rb')
    classifier10 = pickle.load(model10input)
    model10input.close()
    models.append(('rf10',classifier10))

    ensemble = VotingClassifier(estimators=models, voting='soft')
    return ensemble
        
# ###############################################################    
def randomforest_esm_get_models(datapath):

    models = dict()

    modelfile1 = datapath + 'randomforest-esm-all_0.pkl'
    model1input = open(modelfile1,'rb')
    classifier1 = pickle.load(model1input)
    model1input.close()
    models['rf1'] = classifier1

    modelfile2 = datapath + 'randomforest-esm-all_1.pkl'
    model2input = open(modelfile2,'rb')
    classifier2 = pickle.load(model2input)
    model2input.close()
    models['rf2'] = classifier2


    modelfile3 = datapath + 'randomforest-esm-all_2.pkl'
    model3input = open(modelfile3,'rb')
    classifier3 = pickle.load(model3input)
    model3input.close()
    models['rf3'] = classifier3

    modelfile4 = datapath + 'randomforest-esm-all_3.pkl'
    model4input = open(modelfile4,'rb')
    classifier4 = pickle.load(model4input)
    model4input.close()
    models['rf4'] = classifier4

    modelfile5 = datapath + 'randomforest-esm-all_4.pkl'
    model5input = open(modelfile5,'rb')
    classifier5 = pickle.load(model5input)
    model5input.close()
    models['rf5'] = classifier5

    modelfile6 = datapath + 'randomforest-esm-all_5.pkl'
    model6input = open(modelfile6,'rb')
    classifier6 = pickle.load(model6input)
    model6input.close()
    models['rf6'] = classifier6

    modelfile7 = datapath + 'randomforest-esm-all_6.pkl'
    model7input = open(modelfile7,'rb')
    classifier7 = pickle.load(model7input)
    model7input.close()
    models['rf7'] = classifier7

    modelfile8 = datapath + 'randomforest-esm-all_7.pkl'
    model8input = open(modelfile8,'rb')
    classifier8 = pickle.load(model8input)
    model8input.close()
    models['rf8'] = classifier8

    modelfile9 = datapath + 'randomforest-esm-all_8.pkl'
    model9input = open(modelfile9,'rb')
    classifier9 = pickle.load(model9input)
    model9input.close()
    models['rf9'] = classifier9

    modelfile10 = datapath + 'randomforest-esm-all_9.pkl'
    model10input = open(modelfile10,'rb')
    classifier10 = pickle.load(model10input)
    model10input.close()
    models['rf10'] = classifier10

    models['soft_voting'] = randomforest_esm_get_voting(datapath)

    return models
        
# ###############################################################    
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=6)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# ###############################################################    
if  __name__ == "__main__":

    print("get ROC score\n")
