import click
import logging
import os
from pathlib import Path
from data import DataLoad
from utils import Utils
from model import Models
import numpy as np
import matplotlib.pyplot as plt
from Download import downloader
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import random

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42) 
np.seterr(divide='ignore', invalid='ignore')

@click.command()
@click.option("--name",default="Download", prompt="Enter the action to perform [Download, Process, Train, Evaluate, Saliency]", help="This cli helps access the functionality of the program.")
def main(name):
    """Simple script allows to class the necessary routine to execute"""
    actions = ["Download", "Process", "Train", "Saliency"]
    if(name.capitalize() in actions):
        click.echo(f"Run, {name.capitalize()}!")
        if(name.capitalize() == actions[0]):
            """
                This script executes the download routine
            """
            dloader =  downloader()
            dloader.get()
        
        elif(name.capitalize() == actions[1]):
            """
            This section of the code extracts the mri images, bounding boxes, and class labels 
            """
            # Next We call the dataloader method to build the dataset and load it into memory  
            dl =  DataLoad()
            dl.build(flag=True) 
        elif(name.capitalize() == actions[2]):
            """
            This section splits the dataset into train, test, and validation sets. We plot sample figures and save them to the Figures directory
            """
            dl =  DataLoad()
            # dl.build(flag=True)
            dl.load()
            idx = np.random.randint(0, dl.X.shape[0],16)

            Utils.plot_samples(dl.X,dl.y,dl.CLASSES,idx,save=False)

            #@title Splitting the dataset into train/test
            X_train,X_test,y_train,y_test = train_test_split(dl.X_,dl.y, test_size=0.2, shuffle=True)
            Utils.project2D(X_test, y_test,dl.CLASSES,figname='Test_XMRI_TSNE_Before',save=False)
        elif(name.capitalize() == actions[3]):
            """
              Here we perform saliency analysis for the dataset
            """
            pass 
        else:
            click.echo("Invalid option selected, please try again")
    else:
        click.echo("Invalid command select, please try again")
if __name__ == '__main__':
    """
      Here is where all stuff runs
    """
    logging.basicConfig(level=logging.INFO)
    main()

