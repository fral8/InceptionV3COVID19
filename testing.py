
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:24:38 2018

@author: super
"""

import json
from numpy import array

import numpy as np
import tensorflow as tf
import pickle
import os





class Testing():
    
    def __init__(self):

########################################LOAD MODEL################################  
        self.new_model = tf.keras.models.load_model('best/')
        self.new_model.summary()
        x_test = pickle.load(open("x_test.pkl", "rb"))
        y_test = pickle.load(open("y_test.pkl", "rb"))

        self.new_model.evaluate(x_test, y_test, batch_size=16)


    def predict(self,img):
        
        label=["Normale","Pneumonia","COVID"]
        img = np.expand_dims(img, axis=0)
        y=self.new_model.predict(img)
       

        return ([label, y])
    def partial_fit(self,img,label):
        #self.new_model.partial_fit(img,label)
        pass
