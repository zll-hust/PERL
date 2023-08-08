import tensorflow as tf

print("TensorFlow version: ", tf.__version__)
print("Is GPU available: ", tf.test.is_gpu_available())
print("CUDA version: ", tf.sysconfig.get_build_info()['cuda_version'])
print("cuDNN version: ", tf.sysconfig.get_build_info()['cudnn_version'])
