import os
import paddle.fluid as fluid

def load_model(params_file, model_file, use_gpu=False, use_mkl=False, mkl_thread_num=4):
    config = fluid.core.AnalysisConfig(model_file, params_file)

    if use_gpu:
        # 设置GPU初始显存(单位M)和Device ID
        config.enable_use_gpu(100, 0)
    else:
        config.disable_gpu()
    if use_mkl and not use_gpu:
        config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(mkl_thread_num)
    config.disable_glog_info()
    config.enable_memory_optim()

    # 开启计算图分析优化，包括OP融合等
    config.switch_ir_optim(True)
    # 关闭feed和fetch OP使用，使用ZeroCopy接口必须设置此项
    config.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(config)
    return predictor

predictor = load_model('models/paddle/__params__', 'models/paddle/__model__', use_gpu=False)

import numpy as np
data = np.random.rand(1, 3, 260, 260).astype('float32')

input_names = predictor.get_input_names()
input_tensor = predictor.get_input_tensor(input_names[0])
input_tensor.copy_from_cpu(data)

predictor.zero_copy_run()

output_names = predictor.get_output_names()
output_tensor = predictor.get_output_tensor(output_names[0])
result = output_tensor.copy_to_cpu()

print(result.shape)