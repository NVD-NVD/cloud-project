from crnn.predict import predict
import glob
import os

model_default = './model/'
img_path = '/home/zero/Workspace/Courses/Cloud/cloud-project/uploads/'
def NDCV():
    print("NDCV")
    predict_res = predict(img_path)
    print('predict_res in service: ', predict_res)
NDCV()