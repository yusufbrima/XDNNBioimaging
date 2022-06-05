import os
import random
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
import pip
import importlib
import sys
import subprocess

class Utils:
    """ 
        This class impliments utility functions to allow experiment to be carried out successfully. These are custom functions particular to this codebase
    """
    def __init__(self) -> None:
        pass


    @staticmethod
    def resize(x: np.ndarray) -> np.ndarray:
        return tf.keras.layers.experimental.preprocessing.Resizing(225, 225, interpolation='bilinear')(np.expand_dims(x,axis=-1))
    
    @staticmethod
    def plot_samples(X: np.ndarray,y: np.ndarray, Z: np.ndarray, CLASSES: list,idx,figsize=(8,8),save=False):
        """ 
            plot_samples plots randomly selected samples from the dataset
            :X ndarray of the selected mri slices 
            : y ndarray of class labels 0 meningioma, 1 glioma, 2 pituitary tumor
            : CLASSES is a list of the above mentioned class strings 
            :idx ndarray of n sample indices
        """
        fig,axs = plt.subplots(4,4, figsize=figsize)
        fig.subplots_adjust(hspace=.5, wspace=.001)
        for i, ax in zip(list(range(0,len(idx))),axs.ravel()):
            for j in range(Z[idx[i]].shape[0]-1):
                ax.imshow(X[idx[i]])
                ax.scatter(Z[idx[i]][j],  Z[idx[i]][j+1],marker=".", color="red", alpha=0.6)
                ax.set_title(CLASSES[y[idx[i]]])
                ax.set_axis_off()
        plt.show()
        if(save):
            fig.savefig('./Figures/Samples.svg',bbox_inches ="tight",dpi=300)

    @staticmethod
    def project2D(X_,y,CLASSES,figname='XMRI_TSNE',save=False):
        """
            The functions takes a standardized dataset X_ and projects it to 2D using TSNE technique for visualization.
        """
        X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(X_.reshape(X_.shape[0],-1).astype('float64'))
        logging.info("TNSE embedding created successfully. Displaying the 2D projection onto a scatterplot.")
        # #plotting results
        fig = plt.figure(1,figsize=(8,4))
        scatter = plt.scatter(X_embedded[:,0],X_embedded[:,1], c=list(y))
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.legend(handles=scatter.legend_elements()[0], labels=CLASSES,bbox_to_anchor=(1.0, 1.0))
        if(save):
            plt.savefig(f'./Figures/{figname}.svg', bbox_inches ="tight", dpi=300)
            logging.info(f"Figure ./Figures/{figname}.svg written successfully.")
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,CLASSES: list, save=True, filename='confusion_matrix',figsize=(8,6)) -> None:
        """
            : y_true the ground truth class labels
            : y_pred the predicted class labels
            : CLASSES, the list of classes

            The function plots a confusion matrix of the model predictions.
        """
        fig = plt.figure(1,figsize=figsize)
        df_cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index = [i for i in CLASSES],columns = [i for i in CLASSES]) 
        sns.heatmap(df_cm, annot=True, cmap = 'Blues',fmt='g')
        if(save):
            plt.savefig(f'./Figures/{filename}.svg', bbox_inches ="tight", dpi=300)
        plt.close(fig)
        logging.info(f"Plot saved successfully to ./Figures/{filename}.svg")

class PackageManager:
    def __init__(self) -> None:
        pass
    @staticmethod
    def install_and_import(package: str):
        try:
            importlib.import_module(package)
        except ImportError:
            pip.main(['install', package])
        finally:
            globals()[package] = importlib.import_module(package)

if __name__ == "__main__":
    pass
