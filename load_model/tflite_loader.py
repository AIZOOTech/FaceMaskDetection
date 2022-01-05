# -*- encoding=utf-8 -*-
import tflite_runtime.interpreter as tflite

import numpy as np

def load_tflite_model(tflite_model_path):
    '''
    Load the model.
    :param tflite_model_path: model to tflite model.
    :return: interpreter and tensor indexes
    '''
    interpreter = tflite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_idx = input_details[0]['index']
    bbox_idx = output_details[0]['index']
    scores_idx = output_details[1]['index']

    return interpreter, (input_idx, (bbox_idx, scores_idx))


def tflite_inference(interpreter, indexes, img_arr):
    '''
    Receive an image array and run inference
    :param interpreter: tflite interpreter.
    :param indexes: tflite tensor indexes.
    :param img_arr: 3D numpy array, RGB order.
    :return:
    '''
    input_data = np.array(img_arr, dtype=np.float32)
    interpreter.set_tensor(indexes[0], input_data)

    interpreter.invoke()

    bboxes = interpreter.get_tensor(indexes[1][0])
    scores = interpreter.get_tensor(indexes[1][1])

    return bboxes, scores
