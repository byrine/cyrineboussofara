import os
import sys
import datetime
from dataset_preprocessing import training_label_set, training_point_set, validation_label_set,validation_point_set
import pandas as pd
sys.path.insert(0, './')
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from models.sem_seg_model import SEM_SEG_Model


tf.random.set_seed(42)

def train():

	model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])
	
	model.build((config['batch_size'], 3159, 3))
	
	print(model.summary())
	
	model.compile(
		optimizer=keras.optimizers.Adam(config['lr']),
		loss=keras.losses.SparseCategoricalCrossentropy(),
		metrics=[keras.metrics.SparseCategoricalAccuracy()]
	)
	
	
	history=model.fit(
		
		training_point_set,
		training_label_set,
		validation_data=(validation_point_set,validation_label_set),
		validation_steps=10,
		validation_freq=1,
		epochs=100,
		verbose=1
	)
	
	hist_df=pd.DataFrame(history.history)
	training_results='dataset_astyx_hires2019/dataset_astyx_hires2019/data/training_results_astyx.csv'
	with open(training_results,'w') as T:
		hist_df.to_csv(T)

   

if __name__ == '__main__':
       
	train()
