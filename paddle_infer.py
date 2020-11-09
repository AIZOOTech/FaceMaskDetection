import cv2
import argparse
import numpy as np
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
import paddle.fluid as fluid
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
colors = ((0, 255, 0), (0, 0 , 255))

def load_model(model_file, params_file, use_gpu=False, use_mkl=True, mkl_thread_num=4):
    config = fluid.core.AnalysisConfig(model_file, params_file)

    if use_gpu:
        config.enable_use_gpu(100, 0)
    else:
        config.disable_gpu()
    if use_mkl and not use_gpu:
        config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(mkl_thread_num)
    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_ir_optim(True)
    config.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(config)
    return predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--model_dir', type=str, default='models/paddle', help='model path')
    args = parser.parse_args()
    predictor = load_model(args.model_dir+"/__model__",args.model_dir+"/__params__")
    cap = cv2.VideoCapture(0)
    target_shape=(260, 260)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        show = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        image_resized = cv2.resize(img, target_shape)
        image_np = image_resized / 255.0
        image_np = image_np.transpose(2,0,1)
        img = np.expand_dims(image_np,axis=0).copy()
        img = img.astype("float32")
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_tensor(input_names[0])
        input_tensor.copy_from_cpu(img)
        predictor.zero_copy_run()
        output_names = predictor.get_output_names()
        y_bboxes_output = predictor.get_output_tensor(output_names[0])
        y_cls_output = predictor.get_output_tensor(output_names[1])
        y_bboxes_output = y_bboxes_output.copy_to_cpu()
        y_cls_output = y_cls_output.copy_to_cpu()
        y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms.
        keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=0.5, iou_thresh=0.4)
        # keep_idxs  = cv2.dnn.NMSBoxes(y_bboxes.tolist(), bbox_max_scores.tolist(), conf_thresh, iou_thresh)[:,0]
        tl = round(0.002 * (height + width) * 0.5) + 1  # line thickness
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)
            cv2.rectangle(show, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=tl)
            cv2.putText(show, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),3, 0.8, colors[class_id])
        cv2.imshow("img",show)
        cv2.waitKey(1)