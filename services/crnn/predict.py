import os
import tensorflow as tf
from services.crnn.crnn import get_model
from services.crnn.loader import SIZE, MAX_LEN, TextImageGenerator, decode_batch
from keras import backend as K
from keras.preprocessing import image                                                        
import glob
import argparse

def loadmodel(weight_path):
    model = get_model((*SIZE, 3), training=False, finetune=0)
    model.load_weights(weight_path)
    return model

# def predict(img_path):
#     models = glob.glob('./model/best_*.h5')
#     img = image.load_img(img_path, target_size=SIZE[::-1])
#     predict_res = ''
#     for weight_path in models:    
#         print('load {}'.format(weight_path))
#         model = loadmodel(weight_path)
#         y_pred = model.predict(img)
#         decoded_res = decode_batch(y_pred)
#         print('{}: {}'.format(img_path, decoded_res))
        
#         predict_res = predict_res + '\n' + decoded_res
#     print(predict_res)
#     return predict_res

def predict(datapath):
    # sess = tf.Session()
    sess =  tf.compat.v1.Session()
    K.set_session(sess)

    batch_size = 3
    models = glob.glob('{}/best_*.h5'.format('/home/zero/Workspace/Courses/Cloud/cloud-project/services/crnn/model'))
    test_generator  = TextImageGenerator(datapath, None, *SIZE, batch_size, 32, None, False, MAX_LEN)
    test_generator.build_data()
    # decoded_res = []

    for weight_path in models:
        print('load {}'.format(weight_path))
        model = loadmodel(weight_path)
        X_test = test_generator.imgs.transpose((0, 2, 1, 3))
        y_pred = model.predict(X_test, batch_size=3)
        decoded_res = decode_batch(y_pred)
        for i in range(len(test_generator.img_dir)):
            print('{}: {}'.format(test_generator.img_dir[test_generator.indexes[i]], decoded_res[i]))
    return decoded_res

# def predict(model, datapath):
#     sess = tf.Session()
#     K.set_session(sess)

#     batch_size = 3
#     models = glob.glob('{}/best_*.h5'.format(model))
#     test_generator  = TextImageGenerator(datapath, None, *SIZE, batch_size, 32, None, False, MAX_LEN)
#     test_generator.build_data()

#     for weight_path in models:
        
#         print('load {}'.format(weight_path))
#         model = loadmodel(weight_path)
#         X_test = test_generator.imgs.transpose((0, 2, 1, 3))
#         y_pred = model.predict(X_test, batch_size=3)
#         decoded_res = decode_batch(y_pred)
#         for i in range(len(test_generator.img_dir)):
#             print('{}: {}'.format(test_generator.img_dir[test_generator.indexes[i]], decoded_res[i]))

# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default='./model/', type=str)
#     parser.add_argument('--data', default='./data/ocr/preprocess/test/', type=str)
#     parser.add_argument('--device', default=0, type=int)
#     args = parser.parse_args()
    
    
#     os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

#     predict(args.model, args.data)

