import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import h5py
from utils import Utils
import logging

#Data source: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427


class DataLoad:
    """ 
      This class contains utility functions and and properties for managing data access, manipulation and storage.
    
    """
    datapath =  None
    outdir =  None
    CLASSES =  {0 : "meningioma", 1 : "glioma", 2 : " pituitary tumor"}
    
    def __init__(self, datapath = './Data/brainTumorDataPublic',outdir = './Data/brainTumorDataPublic.npz'):
        self.datapath = datapath
        self.outdir = outdir
        logging.info(f'Class instantiated successfully with default input= {self.datapath} and output = {self.outdir} paths set respectively.')
    
    def buildPaths(self):
      """
       : self.datapath returns a list of all files full paths ending with .mat extension
      """
      return [Path(self.datapath, x) for x in os.listdir(self.datapath) if x.endswith('.mat')]

    def build(self, flag=False):

      """
      readData reads the stored .mat files that are passed as a list of strings, preprocess then as ndarray and write then to disk 
      as a compressed npz file
      
      :files is a list of file names that end with .mat extension to be processed
      :outdir is the output path for the compressed numpy array
      """
      if(Path(self.outdir).is_file() and flag == False):
        logging.info("Dataset created already, process exited.")
      else:
        files =  self.buildPaths()
        X = [] #resized images 225x225x1
        y = [] #class labels 0-2
        Z = [] #tumor borders
        A = [] #original images 512x512
        for j in  tqdm(range(len(files))):
            f =  h5py.File(files[j],'r')
            data =  f.get('cjdata/image')
            # if(np.array(data).shape[0] == 512 and np.array(data).shape[1] == 512):
            X.append(np.array(Utils.resize(np.array(data))))
            A.append(np.array(data))
            tumorBorder = np.squeeze(np.array(f.get('cjdata/tumorBorder')))
            Z.append(tumorBorder)
            label =  f.get('cjdata/label')
            label = int(np.squeeze(np.array(label)))
            y.append(label-1)
            f.close()
        np.savez(self.outdir, X = np.array(X), y = np.array(y),Z = np.array(Z,dtype=object),A = np.array(A,dtype=object), C=list(self.CLASSES.values()))
        logging.info("Data has been processed successfully")
    
    def load(self) -> None:
      """
        : indir is the path to the compressed numpy array stored on disk. The function checks if the file exist elese it calls the 
          build method to create the npz file the read from it.
      """
      if(not Path(self.outdir).is_file()):
        logging.warn(f"{self.outdir} not found. Creating new dataset.")
        self.build()
      with  np.load(self.outdir,allow_pickle=True) as data:
        self.X =  data['X']
        self.Z =  data['Z']
        self.A = data['A']
        self.y = data['y']
        self.CLASSES = list(data['C'])
      self.X_ =  self.standardize()
      logging.info(f"{self.X.shape[0]} samples loaded successfully belonging to {len(self.CLASSES)} tumor classes.")
    
    def standardize(self) -> None:
      """
        : self.X that is in R^{B x W x H x C} where B is the batch, W is the width, H is the height and C is the number of channels per image.
        The output is a mean centered transformation of the original images where mu=0, std=1 stored in the variable X_ which has the same dimensions as the original X
      """
      return np.divide((self.X - self.X.mean(axis=0)), self.X.std(axis=0), out=np.zeros_like((self.X - self.X.mean(axis=0))), where=self.X.std(axis=0)!=0)
    
if __name__ == "__main__":
  pass