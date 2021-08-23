import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Average, Input, Concatenate, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121,ResNet50,VGG19,VGG16,InceptionV3,Xception,InceptionResNetV2
from tensorflow.keras.callbacks import CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from keras import layers
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
import csv
import numpy as np
import itertools
from keras.models import load_model

#---------------------------------------------------------------------------------------

traindir = "../input/paperdataset2splitted/tra-val-splitted-seed1337-9-1/train"
valdir = "../input/paperdataset2splitted/tra-val-splitted-seed1337-9-1/val"
testdir = "../input/paperdataset2splitted/test"

batch_fit = 32

datagen=ImageDataGenerator(rescale=1./255)

train_generator=datagen.flow_from_directory(
                directory=traindir,
                batch_size=batch_fit,
                seed=42,
                shuffle=True,
                class_mode= "categorical",
                target_size=(224,224),
                color_mode="rgb")


valid_generator=datagen.flow_from_directory(
                directory=valdir,
                seed=42,
                shuffle=True,
                class_mode= "categorical",
                target_size=(224,224),
                color_mode="rgb")

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
#---------------------------------------------------------------------------------------

a = ([ResNet50,"Resnet50"],[VGG16,"VGG16"],[InceptionV3,"InceptionV3"],[DenseNet121,"DenseNet121"])
b=([VGG19,"VGG19"],[InceptionResNetV2,"InceptionResNetV2"])

#--------------------------------------------------------------------------------------

# softmax the probabilistic predictions
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum() # only difference

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def test_sonuc(model, modelismi):
    
    # ================ data preprocessing =======================
    predicted = model.predict(test_data_gen,verbose = 1)
    label_int_np = test_data_gen.labels
    pred_score_np,label_int_np = predicted,label_int_np
    # define class names
    class_names = ['actinkeratosis','basalcell','carcinoma','derfibroma','melanoma','nevus','pigbenkeratosis','seborkeratosis','vaslesion']
    pred_softmax_np = np.apply_along_axis(lambda x: softmax(x),1,pred_score_np)
    # convert prob result to int result
    pred_int_np = np.argmax(pred_softmax_np, axis=1)
    # ==================== plot counfusion matrix ================
    label_int_np = np.array(label_int_np).T
    cnf_matrix = confusion_matrix(label_int_np, pred_int_np)
    acscore=accuracy_score(label_int_np, pred_int_np)
    print(cnf_matrix)
    print(acscore)
    """
    Plot normalized confusion matrix
    """
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix of model {}'.format(modelismi))
    plt.show()
    # =================== plot ROC curve ================
    """
    Compute ROC curve and ROC area for each class
    """
    enc = OneHotEncoder(sparse=False)
    enc.fit(label_int_np.reshape(-1,1))
    onehot=enc.transform(label_int_np.reshape(-1,1))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    for (i, class_name) in enumerate(class_names):
        fpr[class_name], tpr[class_name], thresholds[class_name] = roc_curve(onehot[:, i], pred_softmax_np[:, i])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])

        youdens = tpr[class_name] - fpr[class_name]
        index=np.argmax(youdens)
        #youden[class_name]=tpr[class_name](index) 
        fpr_val=fpr[class_name][index]
        tpr_val=tpr[class_name][index]
        thresholds_val=thresholds[class_name][index]

        p_auto=pred_softmax_np[:, i].copy()
        t_auto=onehot[:, i].copy()
        p_auto[p_auto>=thresholds_val]=1
        p_auto[p_auto<thresholds_val]=0
        acc=np.float(np.sum(t_auto==p_auto))/t_auto.size


        plt.figure()
        plt.plot(fpr[class_name], tpr[class_name], color='darkorange',
             lw=2, label=class_name+ '(%0.2f)' % roc_auc[class_name])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of %s ' % class_name+ '(AUC={:.2f}, Thr={:.2}, Acc={:.2f}% Model:{}'.format(
                roc_auc[class_name],thresholds_val,acc*100,modelismi))
        plt.savefig('roc_%s.png'%class_name+' of Model: {}'.format(modelismi))
        plt.show()

    # =================== plot PRC curve ================
    """
    Compute PRC curve and recall rate on important points for each class
    """
    from sklearn.metrics import precision_recall_curve
    precision = dict()
    recall = dict()
    for (i, class_name) in enumerate(class_names):
        plt.figure()
        precision[class_name], recall[class_name], _ = precision_recall_curve(onehot[:, i], pred_softmax_np[:, i])
        call95=np.max(recall[class_name][precision[class_name]>=0.95])
        call90=np.max(recall[class_name][precision[class_name]>=0.90])
        call85=np.max(recall[class_name][precision[class_name]>=0.85])
        plt.step(recall[class_name], precision[class_name], color='b', alpha=0.2,
             where='post')
        plt.fill_between(recall[class_name], precision[class_name], step='post', alpha=0.2,
                     color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('PRC of %s ' % class_name + '({:.2f}%, {:.2f}%, {:.2f}%) of Model: {}'.format(call95*100,call90*100,call85*100,modelismi))
        plt.savefig('prc_%s.png'%class_name)
        plt.show()

#---------------------------------------------------------------------------------------
 
input_shape = (224,224,3)
inputs = Input(input_shape)

for sec,modelismi in b:

    try:

        backbone = sec(input_shape=input_shape,include_top=False)
        model = Sequential()
        model.add(backbone)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dropout(0.5))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(9, activation='softmax'))

        model.compile(optimizer= Adam(lr=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
        checkpoint = ModelCheckpoint(('./model_best_weights_{}.h5'.format(modelismi)), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1)

        model.fit_generator(train_generator, epochs=20, steps_per_epoch = 1553, validation_steps = 173 ,validation_data= valid_generator, verbose=1, callbacks=[CSVLogger("./training_logs_{}.csv".format(modelismi),
                                                        append=False,
                                                           separator=';'), early_stop, checkpoint])
        model.save("./{}.h5".format(modelismi))
        test_sonuc(model,modelismi)

    except:
        print("hata var {}".format(modelismi))