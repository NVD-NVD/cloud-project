import os
import tensorflow as tf
from services.crnn.crnn import get_model
from services.crnn.loader import SIZE, MAX_LEN, TextImageGenerator, decode_batch
from keras import backend as K
from keras.preprocessing import image                                                        
import glob

def loadmodel(weight_path):
    model = get_model((*SIZE, 3), training=False, finetune=0)
    model.load_weights(weight_path)
    return model

def predict(datapath):
    sess =  tf.compat.v1.Session()
    K.set_session(sess)

    batch_size = 3
    models = glob.glob('{}/best_*.h5'.format(os.getcwd() + '/services/crnn/model'))
    test_generator  = TextImageGenerator(datapath, None, *SIZE, batch_size, 32, None, False, MAX_LEN)
    test_generator.build_data()

    for weight_path in models:
        print('load {}'.format(weight_path))
        model = loadmodel(weight_path)
        X_test = test_generator.imgs.transpose((0, 2, 1, 3))
        y_pred = model.predict(X_test, batch_size=3)
        decoded_res = decode_batch(y_pred)
        for i in range(len(test_generator.img_dir)):
            print('{}: {}'.format(test_generator.img_dir[test_generator.indexes[i]], decoded_res[i]))
    return decoded_res
