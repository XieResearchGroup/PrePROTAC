import csv
from getdataset import getdataset_iFeature, getdataset_dscript, getdataset_esmFeature, printesmFeature
import torch
import numpy as np
import pandas as pd
import getopt, sys
import shap
import pickle
import matplotlib.pyplot as plt

# ###############################################################    

if  __name__ == "__main__":

    fitting_file = "protein-kinase.list"
    with open(fitting_file) as proteins:
        nfits = len(proteins.readlines())

    print(nfits)
    fit_lable = np.zeros(nfits,dtype=int)

    esmfeature_type = "all"
   
    fit_datapath = "~/preprotac/finalmodel/trainingkinase/data/"

    esmmodel_path = "~/preprotac/finalmodel/10-models/"

    fitting_shape, num_fitting = getdataset_esmFeature(fitting_file,esmfeature_type,fit_lable,fit_datapath, nfits)

    for imodel in range(10):
        modelfile = esmmodel_path + 'randomforest-esm-all-6_' + str(imodel) + '.pkl'
        modelinput = open(modelfile, 'rb')
        classifier = pickle.load(modelinput)
        modelinput.close()

        shap_values = shap.TreeExplainer(classifier).shap_values(fitting_shape)

        f = plt.figure()
        shapfile = 'shap_plot_' + str(imodel)  + '.jpg'
        shap.summary_plot(shap_values,fitting_shape, plot_type="bar", max_display=40)
        f.savefig(shapfile, bbox_inches='tight')

    vals= np.abs(shap_values).mean(0)

    feature_importance = pd.DataFrame(list(zip(fitting_shape.columns(), vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
    print(feature_importance.head())


