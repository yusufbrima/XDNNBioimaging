from curses import flash
import logging
import os
from pathlib import Path
from data import DataLoad
from utils import Utils
import numpy as np
import matplotlib.pyplot as plt
from Download import downloader
class Run:

    def __init__(self) -> None:
        pass



if __name__ == "__main__":
    """
        This script executes all experiments including training and testing the model as well as saliency analysis.
    
    """
    logging.basicConfig(level=logging.INFO)
    dloader =  downloader()
    dloader.get()
    
    
    # Next We call the dataloader method to build the dataset and load it into memory  
    # dl =  DataLoad()
    # dl.build(flag=True)
    # dl.load()
    # idx = np.random.randint(0, dl.X.shape[0],16)

    # Utils.plot_samples(dl.X,dl.y,dl.CLASSES,idx,save=True)
    # Utils.plot_samp(dl.A,dl.y,dl.Z,dl.CLASSES,idx,save=True)
    # (X: np.ndarray, y: np.ndarray, Z: np.ndarray, CLASSES: list,idx,figsize=(8,8),save=False)
    # Utils.project2D(dl.X_, dl.y,dl.CLASSES,figname='XMRI_TSNE_Before',save=False)
    
