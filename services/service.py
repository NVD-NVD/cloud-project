from tensorflow.keras import models
from keras.preprocessing import image
import numpy as np

from keras import applications
from keras.layers import Input, Reshape, Dense, Dropout, Bidirectional, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.merge import add, concatenate
# from tensorflow.keras.layers import add, concatenate
from keras.models import Model, save_model, load_model
import numpy as np
from keras import backend as K
import glob

letters = " #'()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdeghiklmnopqrstuvxwyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
MAX_LEN = 70
SIZE = 2560, 160
CHAR_DICT = len(letters) + 1

def NDCV():
    # model_weights = "digit_weight.h5"
    model_path = './model/best_0.h5'
    model = load_model(model_path, compile = True)

    img = image.load_img("./model/1.jpg")
    img = np.expand_dims(img, axis=0)
    predict = model.predict(img)
    print(predict)
    return "nhan dang chu viet"

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(input_shape, training, finetune):
    inputs = Input(name='the_inputs', shape=input_shape, dtype='float32')
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    inner = base_model(inputs)
    inner = Reshape(target_shape=(int(inner.shape[1]), -1), name='reshape')(inner)
    inner = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 
    inner = Dropout(0.25)(inner) 
    lstm = Bidirectional(LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.25, recurrent_dropout=0.25))(inner) 

    y_pred = Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal',name='dense2')(lstm)
    
    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    for layer in base_model.layers:
        layer.trainable = finetune
    
    y_func = K.function([inputs], [y_pred])
    
    if training:
        Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out).summary()
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out), y_func
    else:
        return Model(inputs=[inputs], outputs=y_pred)


NDCV()