if [ ! -d models/paddle ]; then
    x2paddle --framework=caffe --prototxt=models/face_mask_detection.prototxt --weight=models/face_mask_detection.caffemodel --save_dir=./ --params_merge
    mv inference_model models/paddle
fi
python3 paddle_infer.py