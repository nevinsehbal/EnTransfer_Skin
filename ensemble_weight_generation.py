import os
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
​
model1 = load_model('../input/train-loop-for-makale/model_best_weights_VGG19.h5', compile = True)
model2 = load_model('../input/train-loop-for-makale/model_best_weights_InceptionResNetV2.h5', compile = True)
​
valdir = "../input/paperdataset2splitted/tra-val-splitted-seed1337-9-1/val"
testdir = "../input/paperdataset2splitted/test"
​
batch_fit = 32
datagen=ImageDataGenerator(rescale=1./255)
valid_generator=datagen.flow_from_directory(
                                           directory=valdir, 
                                           target_size=(224, 224),
                                           batch_size = 1,
                                           color_mode="rgb",
                                           class_mode = None,#'categorical',
                                           shuffle = False,
                                           seed = 42,
                                            )
​
test_image_generator = ImageDataGenerator(rescale=1./255)
test_data_gen = test_image_generator.flow_from_directory(
                                                       directory = testdir, 
                                                       target_size=(224, 224),
                                                       batch_size = 1,
                                                       color_mode="rgb",
                                                       class_mode = None,#'categorical',
                                                       shuffle = False,
                                                       seed = 42,
                                                        )
#----------------------------------------------------------------------------------------------------

predicted1 = model1.predict(valid_generator,verbose = 1)
predicted2 = model2.predict(valid_generator,verbose = 1)

label_int_np = valid_generator.labels

pred_score1_np = predicted1
pred_score2_np = predicted2

label_int_np2 = test_data_gen.labels

import numpy as np 

def softmax(x): 
    """Compute softmax values for each sets of scores in x.""" 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum() # only difference 

pred1_softmax_np = np.apply_along_axis(lambda x: softmax(x),1,pred_score1_np) 
pred2_softmax_np = np.apply_along_axis(lambda x: softmax(x),1,pred_score2_np) 

onehot_encoded = list()
for value in label_int_np:
    letter = [0 for _ in range(0,9)]
    letter[value] = 1
    onehot_encoded.append(letter)

label_int_arr = np.array(onehot_encoded)

def sigmafinder(arrorig,arrpred):
    sigma=np.abs(np.subtract(arrorig,arrpred))
    sigma = np.sum(sigma)
    sigma = sigma
    #print(sigma)
    return sigma

sigma1 = sigmafinder(label_int_arr,pred1_softmax_np)
sigma2 = sigmafinder(label_int_arr,pred2_softmax_np)

k = (sigma1**-1 + sigma2**-1)**-1
w1= k/sigma1
w2 = k/sigma2

Y = w1*pred1_softmax_np + w2*pred2_softmax_np
