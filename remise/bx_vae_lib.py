#!/usr/bin/env python
# coding: utf-8

# In[1]:

## sources sur les autoencoders:
## ->https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
## ->https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
## ->https://www.jeremyjordan.me/variational-autoencoders/
## ->https://www.tensorflow.org/api_docs 
## ->https://www.tensorflow.org/probability/examples/A_Tour_of_TensorFlow_Probability

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from  IPython import display
import numpy as np
import matplotlib.pyplot as plt 
import os
import pathlib
import shutil
import tempfile
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfpd = tfp.distributions


class VAE():
    def __init__(self, input_shape,lat_s_shape, weight = 0.00005,encoder = False, decoder = False):
        self.input_shape = input_shape
        self.weight = weight
        self.lat_s_shape = lat_s_shape
        self.prior = tfpd.Independent(tfpd.Normal(loc=tf.zeros(self.lat_s_shape), scale=.5),
                                      reinterpreted_batch_ndims=1) 
        self.encoder_m = None
        self.decoder_m = None
        self.vae = None
        
        self.one_dim = 1 
        for elements in input_shape:
            self.one_dim *= elements
        
        ## vanilla encoder
        self.encoder_l = [
            keras.Input(shape = self.input_shape),
            
            layers.Flatten(),
            layers.Dense(tfp.layers.IndependentNormal.params_size(self.lat_s_shape)),
            tfp.layers.IndependentNormal(self.lat_s_shape, 
            convert_to_tensor_fn=tfpd.Distribution.sample, 
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(self.prior, weight=self.weight), 
            name='lat_space')
        ]
        
        if encoder != False:  
            self.encoder_l = self.encoder_l[:1] + encoder + self.encoder_l[1:]
        
        self.decoder_l = [
            keras.Input(shape = (self.lat_s_shape,)),
            
            layers.Flatten(),
            layers.Dense(self.one_dim),
            layers.Reshape(self.input_shape)
        ]
        
        if decoder != False:  
            self.decoder_l = self.decoder_l[:1] + decoder + self.decoder_l[1:]
            
            
    def build_encoder(self):
        encoder_input = self.encoder_l[0]
        x = self.encoder_l[1](encoder_input)

        for lays1 in self.encoder_l[2:-1]:
            x = lays1(x)

        out_encoder = self.encoder_l[-1](x)
        self.encoder_m = keras.Model(encoder_input, out_encoder)
        
    def build_decoder(self):
        decoder_input = self.decoder_l[0]
        x = self.decoder_l[1](decoder_input)

        for lays2 in self.decoder_l[2:-1]:
            x = lays2(x)

        decoder_out = self.decoder_l[-1](x)
        self.decoder_m = keras.Model(decoder_input, decoder_out)    
        
    def build_vae(self):
        auto_in = keras.Input(shape = self.input_shape)
        auto_e = self.encoder_m(auto_in)
        auto_d  = self.decoder_m(auto_e)
        self.vae = keras.Model(auto_in, auto_d)

"""

MIT License

Copyright (c) 2022 billy xue

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

