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
import logging

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
    def plot_samples(X: np.ndarray,y: np.ndarray, Z: np.ndarray, CLASSES: list,idx,figsize=(8,8),save=False, show=True):
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
        if (show):
            plt.show()
        if(save):
            fig.savefig('./Figures/Samples.svg',bbox_inches ="tight",dpi=300)

    @staticmethod
    def project2D(X_,y,CLASSES,figname='XMRI_TSNE',save=False, show=True):
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
        if(show):
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
    
    @staticmethod
    def plot_evaluation(dft, save=True, filename="Train_Time_Accuracy.svg"):
        #@title Plotting Models test performance
        # sort df by Count column
        dft['model'] = dft['model'].apply(lambda x: str(x).title())
        pd_df = dft.sort_values(['f1_score']).reset_index(drop=True)
        plt.figure(figsize=(8,6))
        # plot barh chart with index as x values
        ax = sns.barplot(pd_df.model, pd_df.f1_score)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(np.round(float(x),1))))
        ax.set_xlabel("Model Architecture",fontsize=12)
        ax.set_ylabel(r"$F_1$ Score",fontsize=12)
        # add proper Dim values as x labels
        ax.set_xticklabels(pd_df.model)
        for item in ax.get_xticklabels(): item.set_rotation(90)
        for i, v in enumerate(pd_df["f1_score"].iteritems()):        
            ax.text(i ,v[1], "{:,}".format(np.round(v[1],2)), color='b', va ='bottom', rotation=40, fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./Figures/{filename}', bbox_inches ="tight", dpi=300)
        plt.show()
    
    @staticmethod
    def create_embedding(model_name: str, X_data: np.ndarray):
        logging.info(f"Loading {model_name.capitalize()} model from ./Models/{model_name}")
        model = tf.keras.models.load_model(f"./Models/{model_name}",compile=True)
        repmodel  = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output, name = model.name)
        logging.info("Model loaded and re-initialized successfully")
        logging.info(f"Starting inference on {X_data.shape[0]} samples")
        X_hat =  repmodel.predict(X_data)
        logging.info("Inference completed successfully")
        return X_hat
    
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
    logging.basicConfig(level=logging.INFO)
