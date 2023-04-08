import os
import sys
import datetime
import pandas as pd
sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from models.sem_seg_model import SEM_SEG_Model
from models.sem_seg_msg_model import SEM_SEG_MSG_Model

tf.random.set_seed(42)

'''
for the loading of TFRecord:
	-create a tf.data.TFRecordDataset 
	-create a feature description and fix the length of the features
	-parse the serialized example
	-assign each parsed serialized example value to the feature created in the description
'''	 
def load_dataset(in_file,batch_size):
	n_points=8192
	assert os.path.isfile(in_file), '[error] dataset path not found'
	n_points = n_points
	shuffle_buffer = 1000
	def parse_function(in_file):
		features = {
                'labels': tf.io.FixedLenFeature([n_points], dtype=tf.int64),
                'points': tf.io.FixedLenFeature([n_points*3], dtype=tf.float32)
        }
		example = tf.io.parse_single_example(in_file, features)
		labels=example['labels'] 
		points=example['points']
		points = tf.reshape(points, (n_points, 3))
		labels = tf.reshape(labels, (n_points, 1))
		shuffle_idx = tf.range(points.shape[0])
		shuffle_idx = tf.random.shuffle(shuffle_idx)
		points = tf.gather(points, shuffle_idx)
		labels = tf.gather(labels, shuffle_idx)
		return points,labels
		
	dataset = tf.data.TFRecordDataset(in_file)
	#obtaining new dataset samples from every shuffle_buffer labels or points
	dataset = dataset.shuffle(shuffle_buffer)
	#implementing the parse function to each element of the data (labels and points)
	dataset = dataset.map(parse_function)
	#precising the batch size
	dataset = dataset.batch(batch_size, drop_remainder=True)


	return dataset

'''
tests:

--1st Test: training for 100 epochs
--2nd Test: training for 1000 epochs
-- The remaining tests with MSG modules carried out for 100 epochs
'''


def train():
	config = {
		'train_ds' : 'data/scannet_train.tfrecord',
		'val_ds' : 'data/scannet_val.tfrecord',
		
		'log_freq' : 10,
		'test_freq' : 100,
		'batch_size' : 4,
		'num_classes' : 21,
		'lr' : 0.001,
		'bn' : False,
	}

	model = SEM_SEG_MSG_Model(config['batch_size'], config['num_classes'], config['bn'])

	train_ds = load_dataset(config['train_ds'], config['batch_size'])
	val_ds = load_dataset(config['val_ds'], config['batch_size'])
	
	
	model.build((config['batch_size'], 8192, 3))
	print(model.summary())

	model.compile(
		optimizer=keras.optimizers.Adam(config['lr']),
		loss=keras.losses.SparseCategoricalCrossentropy(),
		metrics=[keras.metrics.SparseCategoricalAccuracy()]
	)

	history=model.fit(
		train_ds,
		validation_data=val_ds,
		validation_steps=10,
		validation_freq=1,
		epochs=100,
		verbose=1
	)

	#Just saving the history of the trainings
	hist_df=pd.DataFrame(history.history)
	training_results='data/training_results_scan_100_msg_newradiusandnboflayers.csv'
	with open(training_results,'w') as T:
		hist_df.to_csv(T)

if __name__ == '__main__':



	train()
