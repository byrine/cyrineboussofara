import numpy as np
import os
import random 
import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import math
import tensorflow as tf

files = []


SOURCE='dataset_astyx_hires2019/dataset_astyx_hires2019/radar_6455/'


#list containing the 9 classes present in the dataset
CLASS_NAMES=['Car','Truck','Person','Bus','Trailer','Cyclist','Motorcyclist',"Towed Object", "Other Vehicle"]


orientation_quat = []
labels = []
GroundTruth='dataset_astyx_hires2019/dataset_astyx_hires2019/groundtruth_obj3d/'

for filename in os.listdir(GroundTruth):
   filename= GroundTruth + filename
    with open(filename) as jsondata:
        d = json.load(jsondata)
    #list of the quaternions of each oject and list of the classes of the objects    
    for x in d["objects"]:
        orientation_quat.append(x["orientation_quat"])
        labels.append(x["classname"])


flat_list= []
#flattened list of the individual components of the quaternions of each object all together
for sublist in orientation_quat:
    for item in sublist:
        flat_list.append(item)


orientationid_to_pointid= {}

#dictionary containing the individual components of the quaternions as key and their indexes in the flattened list as values
for i in range(len(flat_list)):
    if flat_list[i] not in orientationid_to_pointid:
        orientationid_to_pointid[flat_list[i]] = []
    orientationid_to_pointid[flat_list[i]].append(i)

#import pdb; pdb.set_trace()

#gathering all the files .txt in one list
for filename in os.listdir(SOURCE):
    file = SOURCE + filename
    if os.path.getsize(file) > 0:
        files.append(filename)
    else:
        print(filename + " is zero length, so ignoring.")

cleaned_list=[]

#cleaning the files present in the previous list (skip the header and take the first 3 columns only)
for i in range(len(files)):
    my_file=files[i]
    my_array= np.genfromtxt(SOURCE+my_file, skip_header=2,)
    my_array= my_array[:,:-2]
    cleaned_list.append(my_array)

#counting the number of objects for each class and plotting the corresponding results
class_counts = Counter(labels)
df = pd.DataFrame.from_dict(class_counts, orient='index')
df.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Distibution')
plt.title('Distribution of classes')
#plt.show()


# Each instance's points
instance_points_list = []
semantic_labels_list = []
points=[]
#import pdb; pdb.set_trace()



        
#import pdb; pdb.set_trace()


for i in range(len(cleaned_list)):

    file_df=cleaned_list[i]
    for x,y,z in file_df:
        point= [x, y, z]
        
        #import pdb; pdb.set_trace()        

        #list of the points (xyz coordinates)of each object
        points.append(point)
#import pdb; pdb.set_trace()

#the above list converted to an array of type float
points=np.array(points, dtype='float32')  

#import pdb; pdb.set_trace()



for i in range(len(orientation_quat)):
    orientation_ids =orientation_quat[i]
    point_ids =[]
    for element in orientation_ids:
        #indexes of the different components of each quaternion of each object inside the dictionary 
        point_ids += orientationid_to_pointid[element]
    #from these indexes, selecting the points corresponding to each object    
    instance_points = points[np.array(point_ids),:]
    #taking the first 4 arrays of the arrays of points only 
    instance_points=instance_points[:4,:]
    instance_points_list.append(instance_points)
#import pdb; pdb.set_trace()
    
    label=labels[i]
    #searching for the index of each object's class inside the classname list
    label = CLASS_NAMES.index(label)
    #appending the indexes(in a ones array with the same number of rows as the points corresponding to each object and one column)
    semantic_labels_list.append((np.ones((instance_points.shape[0], 1))*label).astype('int64'))
#import pdb; pdb.set_trace()

#converting the lists of points and indexes' labels to numpy array and casting them to the specific type
obj_points = np.array( instance_points_list).astype('float32')
semantic_labels = np.array( semantic_labels_list).astype('int64')





#import pdb; pdb.set_trace()
'''
training set: 80% of the data
#2 training arrays & validation arrays: 
-> split_name_label_set: containing label indexes 
-> split_name_point_set: containing the points 
#reshaping the resulting sets to have a tensor of shape [batch_size, number_of_points, dimension]
-> The batch_size is 4 
-> Number of points depends on training or validation set (here number of points for training is 2527 and for the validation 632)
-> Dimension either 3 for points sets or 1 for label sets
'''
SPLIT_SIZE= 0.8

training_label_length = int(len(obj_points)* SPLIT_SIZE)
validation_label_length = int(len(semantic_labels) - training_label_length)

#shuffling the data 

np.random.shuffle(semantic_labels)

training_label_set=semantic_labels[0:training_label_length]
training_label_set=tf.reshape(training_label_set,(4,2527,1))


validation_label_set=semantic_labels[-validation_label_length:]
validation_label_set=tf.reshape(validation_label_set,(4,632,1))

training_point_length = int( len(obj_points)* SPLIT_SIZE)
validation_point_length = int(len(obj_points)- training_point_length)

np.random.shuffle(obj_points)

training_point_set=obj_points[0:training_point_length]
training_point_set=tf.reshape(training_point_set,(4,2527,3))

validation_point_set=obj_points[-validation_point_length:]
validation_point_set=tf.reshape(validation_point_set,(4,632,3))

#import pdb; pdb.set_trace()


