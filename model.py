import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score
from tensorflow.keras.applications import VGG16,VGG19, ResNet50,InceptionV3,Xception,ResNet50V2,DenseNet121,EfficientNetB0 #EfficientNetV2B0
import logging
import time
from utils import Utils
import pandas as pd
import numpy as np
from tqdm import tqdm

class Models:
    """
      This class implements keras model construction methods and properties for training and inference.
    
    """
    MODELS = [VGG16,VGG19] #, ResNet50,Xception,ResNet50V2,InceptionV3,DenseNet121,EfficientNetB0
    
    def __init__(self):
        pass
    def build(self,Model,input_shape, num_classes):
        K.clear_session() # Clear previous models from memory.
        base_model = Model(weights=None, include_top=False, input_shape=input_shape)
        x = base_model.output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs=base_model.input, outputs=x, name = base_model.name)
        optim_params = dict(learning_rate = 0.001,momentum = 0.9394867962846013,decay = 0.0003)
        model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(**optim_params),metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
        logging.info(f"Model built successfully {model.name}")
        return model
    
    def train(self,X_train: np.ndarray,X_test: np.ndarray,y_train: np.ndarray,y_test: np.ndarray,input_shape: tuple,output_nums: int,CLASSES: list,n=10):
        """
        This method trains a set of models and evalute them respective. The evaluation results are written to a csv file.
        
        """
        hist = {'model':[], 'accuracy': [],'f1_score':[], 'time':[]}
        for j in  tqdm(range(len(self.MODELS))):
            model =  self.build(self.MODELS[j], input_shape, output_nums)
            logging.info(f"We are training {model.name}")
            logging.info("====================================================================")
            start = time.time()
            _ =  model.fit(x = X_train, y =y_train, epochs=n, validation_split = 0.1 )
            stop = time.time()
            y_pred = np.argmax(model.predict(X_test), axis = 1)
            hist['model'].append(model.name)
            hist['accuracy'].append(accuracy_score(y_test, y_pred))
            hist['f1_score'].append(f1_score(y_test, y_pred, average='micro'))
            hist['time'].append(stop - start)
            Utils.plot_confusion_matrix(y_test, y_pred,CLASSES, save=True,filename=f"{model.name}_plot_confusion_matrix.png" )
            model.save(f'./Models/{model.name}')
            logging.info(f"{model.name} trained and evaluted successfully.")
        pd.DataFrame(hist).to_csv('./Data/Evaluation_Results.csv')
        logging.info("Evaluation results written to ./Data/Evaluation_Results.csv")
        return pd.DataFrame(hist)


if __name__ == "__main__":
    pass