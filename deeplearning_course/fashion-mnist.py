import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

#--------------- Read/prep Data ---------------#
img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw, train_size, val_size):
    """ 
    Prepare the raw data to be used by our model. 

    Parameters:
    raw (nparray): raw data which will be prepped
    train_size (int): size of training 
    val_size (int): size of values
    
    Returns:
    nparray: prepped data to be used in training
    """
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data, train_size=50000, val_size=5000)

#--------------- Specify Model w/o Strides & Dropout ---------------#
def plain_model():
    """
    Pre optimization model.
    This model won't be used for our final predictions. 
    """
    fashion_model = Sequential()
    fashion_model.add(Conv2D(12, kernel_size=(3,3),
                    activation='relu',
                    input_shape=(img_rows, img_cols, 1)))
    fashion_model.add(Conv2D(12, kernel_size=(3,3), activation='relu'))
    fashion_model.add(Conv2D(12, kernel_size=(3,3), activation='relu'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(100, activation='relu'))
    fashion_model.add(Dense(num_classes, activation='softmax'))
    return fashion_model

#--------------- Specify Model w/ Dropout ---------------#
def upgraded_model():
    """
    Upgraded model which includes dropouts. 
    This model will be used for final predictions. 

    Returns:
    Built model to be trained. 
    """
    fashion_model_upg = Sequential()
    fashion_model_upg.add(Conv2D(12, kernel_size=(3,3),
                                strides=2,
                                activation='relu',
                                input_shape=(img_rows, img_cols, 1)))
    fashion_model_upg.add(Dropout(.5))
    fashion_model_upg.add(Conv2D(12, kernel_size=(3,3), activation='relu'))
    fashion_model_upg.add(Dropout(.5))
    fashion_model_upg.add(Conv2D(12, kernel_size=(3,3), activation='relu'))
    fashion_model_upg.add(Dropout(.5))
    fashion_model_upg.add(Flatten())
    fashion_model_upg.add(Dense(100, activation='relu'))
    fashion_model_upg.add(Dense(num_classes, activation='softmax'))
    return fashion_model_upg

#--------------- Compile/fit Model ---------------#
model = upgraded_model()
model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer='adam',
                            metrics=['accuracy'])
model.fit(x,y,batch_size=100, epochs=4, validation_split=0.2)

#--------------- Omit submission code ---------------#
