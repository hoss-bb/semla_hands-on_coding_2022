# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from abc import ABC, abstractmethod
from random import shuffle
import numpy as np
from scipy.special import softmax
import tensorflow as tf
import os
import glob
from os import path, strerror
import errno

class FeatureEngineering(ABC):

    @abstractmethod
    def preprocess(self, data):
        pass

class Model:

    def __init__(self, model_path=None, feature_engineering=None, batch_size=None):
        if model_path != None:
            self.load(model_path)
        self.feature_engineering = feature_engineering
        self.batch_size = batch_size

    @abstractmethod
    def load(self, model_path):
        pass

    @abstractmethod    
    def predict(self, data):
        pass

class Batch:
    
    def __init__(self, idx, inputs, labels):
        self.idx = idx
        self.inputs = inputs
        self.labels = labels
        self.predictions = None
        self.probabilities = None
    
    def set_predictions(self, predictions):
        self.predictions = predictions
    
    def set_probabilities(self, probabilities):
        self.probabilities = probabilities

def load_data(persistence_path):
  if not path.exists(persistence_path):
    raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), persistence_path)
  batches = []
  for fpath in glob.iglob(os.path.join(persistence_path, '*.npz')):
    _, tail = os.path.split(fpath)
    idx = int(tail.split('.')[0])
    b_info = np.load(fpath)
    batch = Batch(idx, b_info['inputs'], b_info['labels'])
    batches.append(batch)
  return batches

class Lambda(FeatureEngineering):

    def __init__(self, func):
        self.func = func
    
    def preprocess(self, data):
        return self.func(data)

class TFClassifier(Model):

    def __init__(self, model_path=None, feature_engineering=None, batch_size=None):
        super().__init__(model_path, feature_engineering, batch_size)
    
    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
    
    def set_model(self, model):
        self.model = model
        return self
        
    def predict(self, data):
        if self.feature_engineering:
            data = self.feature_engineering.preprocess(data)
        if self.batch_size != None:
            outputs = np.vstack([self.model.predict(data[pos:pos+self.batch_size]) 
                                for pos in range(0, len(data), self.batch_size)])
        else:
            outputs = self.model.predict(data)
        if np.isclose(np.max(np.sum(outputs, axis=1)), 1.):
            probabilities = outputs
        else:
            probabilities = softmax(outputs, axis=1)
        predictions = np.argmax(probabilities, axis=1) 
        return predictions, probabilities
        