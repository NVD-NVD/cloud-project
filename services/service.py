from services.crnn.predict import predict
import glob
import os

model_default = './model/'
img_path = os.getcwd() + '/uploads/'
def NDCV():
    print("NDCV")
    predict_res = predict(img_path)
    print('predict_res in service: ', predict_res)
    return predict_res
# NDCV()