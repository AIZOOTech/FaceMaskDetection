import caffe
import numpy as np

def load_caffe_model(prototxt_path, caffemodel_path):
    model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    return model

def caffe_inference(model, img_arr):
    model.blobs['data'].data[...] = img_arr
    result = model.forward() # 输出四个分支
    y_bboxes = result['loc_branch_concat']
    y_scores = result['cls_branch_concat']
    return y_bboxes, y_scores