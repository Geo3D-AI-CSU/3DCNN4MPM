# encoding:gbk
from __future__ import division
import os
import csv
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.model_selection import KFold
import time
import datetime

'''组织数据格式'''
def change_format(data, batchsize, deep, height, weight, channel, mark):
	
	data = np.array(data)
	if mark == 'TRUE':
		outdata = data.reshape(batchsize, deep, height, weight, channel)
		#将一维数组变成n维数组
		#张量转数组，列表不能转张量，否则后续输入会有问题
		
	elif mark == 'FALSE':
		outdata = data.reshape(batchsize, deep, height, weight)
		#将一维数组变成n维张量
	
	return outdata

'''读取csv数据'''
'''
逐像元依次读取特征即可，NDWHC括号由外向内的个数。
eg.[a1,a2,a3,b1,b2,b3,c1,c2,c3]
N:batch_size; D:K; W:J; H:I; C:Feature
'''
def read_csv(filepath, filenames, batchsize, deep, height, weight, channel):
	
	data = []
	label = []
	
	#open()读取的是路径加文件名的数据，因此需要将路径作为参数传入进行一个拼接
	# for name in filenames:
	#
	# 	csvfile = open(os.path.join(filepath, name), 'r')
	# 	reader = csv.reader(csvfile)
	# 	for item in reader:
	# 		if reader.line_num == 1: #第一行为属性名
	# 			continue
	# 		#data.append(item[3:-2]) #前三列为I,J,K; 后两列为标签
	# 		#[3:-2]是指从第4列到倒数第三列
	# 		a = np.array(item[3:10])
	# 		#b = np.array(item[16:20])
	# 		#data.append(np.concatenate((a,b)).tolist())
	# 		data.append(a.tolist())
	# 		label.append(item[-1]) #后两列为标签，读取最后一列为非稀疏
	#
	# data = np.array(data)
	# label = np.array(label)
	# data = data.flatten() #将多维数组转换成一维数组
	# data = data.astype(np.float) #将字符串转化为float
	# label = label.astype(np.int)
	# data = data.tolist() #数组转列表
	# label = label.tolist()
	# outdata = change_format(data, batchsize, deep, height, weight, channel, 'TRUE') #列表转多维数组
	# label = change_format(label, batchsize, deep, height, weight, channel, 'FALSE') #列表转多维数组，非稀疏
	#
	# return outdata, label
	for name in filenames:

		csvfile = open(os.path.join(filepath, name), 'r')
		reader = csv.reader(csvfile)
		for item in reader:
			if reader.line_num == 1:  # 第一行为属性名
				continue
			# data.append(item[3:-2]) #前三列为I,J,K; 后两列为标签
			# [3:-2]是指从第4列到倒数第三列
			a = np.array(item[3:10])
			# b = np.array(item[16:20])
			# data.append(np.concatenate((a,b)).tolist())
			data.append(a.tolist())
			label.append(item[-1])  # 后两列为标签，读取最后一列为非稀疏

	data = np.array(data)
	label = np.array(label)
	data = data.flatten()  # 将多维数组转换成一维数组
	data = data.astype(np.float)  # 将字符串转化为float
	label = label.astype(np.int)
	data = data.tolist()  # 数组转列表
	label = label.tolist()
	outdata = change_format(data, batchsize, deep, height, weight, channel, 'TRUE')  # 列表转多维数组
	label = change_format(label, batchsize, deep, height, weight, channel, 'FALSE')  # 列表转多维数组，非稀疏

	return outdata, label
'''k折交叉验证'''
def cross_validation(filelist, fold):
	'''
	input: filelist输入为文件名列表，数组形式
	output: 每个类别包含的索引
	StratifiedKFold方法会根据类别（y数据）对数据（x数据）进行分层抽样，因为没有y数据，因此不采用该方法。
	此处采用的是KFold方法。
	'''
	
	filelist = np.array(filelist)
	
	kf = KFold(n_splits = fold, random_state=2020, shuffle=True)
	#random_state: 随机数种子，只有在shuffle为TRUE时生效
	kf.get_n_splits(filelist) # 查询分成几个组
	
	train_list = []
	test_list = []
	
	#划分后传出的是索引,将索引转换为数据列表
	for train_index, test_index in kf.split(filelist):
		X_train,X_test = filelist[train_index],filelist[test_index]
		train_list.append(X_train.tolist())
		test_list.append(X_test.tolist())
	
	return train_list, test_list
	
'''
将十折交叉验证的结果文件名存储到txt中
'''
	
'''占位符'''
def placeholder_inputs(batch_size, size, channel):
	
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, size, size, size, channel))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, size, size, size)) #非稀疏
    
    return pointclouds_pl, labels_pl

'''输出'''
def log_string(LOG_FOUT, out_str):
	
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
    
'''计算学习率'''
def get_learning_rate(base_learning_rate, 
                      batch, batch_size,
                      decay_step,
                      decay_rate):
						  
    learning_rate = tf.train.exponential_decay(
                        base_learning_rate,  # Base learning rate.
                        batch * batch_size,  # Current index into the dataset.
                        decay_step,          # Decay step.
                        decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    
    return learning_rate        

'''计算衰减率'''
def get_bn_decay(bn_init_decay,
                 batch, batch_size,
                 bn_decay_decay_step,
                 bn_decay_decay_rate,
                 bn_decay_clip):
					 
    bn_momentum = tf.train.exponential_decay(
                      bn_init_decay,
                      batch * batch_size,
                      bn_decay_decay_step,
                      bn_decay_decay_rate,
                      staircase=True)
    bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)
    
    return bn_decay

'''
进行批量归一化:
实现了在神经网络层的中间进行预处理的操作，即在上一层的输入归一化处理后再进入网络的下一层，
这样可有效地防止“梯度弥散”，加速网络训练，加快收敛。
可以减少甚至取消DropOut的使用。
'''
def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond.
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed
  
'''三维卷积归一化'''
def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 3D convolutional maps.
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)

'''全连接归一化'''
def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
  """ Batch normalization on FC data.
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)

'''y=wx+b, w为weight_variable, b为bias_variable'''
'''初始化过滤器, w与输入图像相关'''
def weight_variable(shape, seed=0):
	
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed) #初始化赋值,符合正态分布,标准差为0.1
    w = tf.Variable(initial)
    
    return w
 
'''初始化偏置, b与输出图像相关'''
def bias_variable(shape):
	
    initial = tf.constant(0.1, shape=shape) #初始化bias,用0.1填充
    b = tf.Variable(initial)
    
    return b
    
'''三维卷积'''
'''
data_format：可选的string,可以是："NDHWC", "NCDHW"；默认为"NDHWC".输入和输出数据的数据格式；
使用默认格式"NDHWC",数据按以下顺序存储：[batch,in_depth,in_height,in_width,in_channels]；
或者,格式可以是“NCDHW”,数据存储顺序是：[batch,in_channels,in_depth,in_height,in_width]。
输入数据的类型为dtype = tf.float32,否则可能会出错。
'''
def conv3D(inputs, 
           kernel_size, input_channels, output_channels,
           strides, 
           padding = 'SAME',
           seed = 0,
           dp = False,
           bn = False,
           bn_decay = None,
           is_training = None):
	
	kernel_d, kernel_h, kernel_w = kernel_size
	kernel_shape = [kernel_d, kernel_h, kernel_w, input_channels, output_channels]
	w = weight_variable(kernel_shape, seed)
	
	b = bias_variable([output_channels])
	
	stride_d, stride_h, stride_w = strides
	stride = [1, stride_d, stride_h, stride_w, 1]
	
	result = tf.nn.bias_add(tf.nn.conv3d(inputs, w, stride, padding = padding), b)
	
	if dp:
		result = tf.nn.dropout(result, 0.8)
	
	if bn:
		result = batch_norm_for_conv3d(result, is_training, bn_decay=bn_decay, scope='bn')
	
	result = tf.nn.relu(result)
	
	return result
'''三维全连接'''
def fully_connected_layer(inputs, num_units, is_training, bn_decay=None, activation_fn=tf.nn.relu):
    """
    Create a fully connected layer for the 3D CNN model.

    Args:
        inputs: The input tensor to the fully connected layer.
        num_units: The number of units (neurons) in the fully connected layer.
        is_training: A boolean placeholder indicating whether the model is in training mode.
        bn_decay: Batch normalization decay (if using batch normalization).
        activation_fn: The activation function to apply after the fully connected layer (default is ReLU).

    Returns:
        fc: The output tensor of the fully connected layer.
    """
    with tf.variable_scope('fully_connected', reuse=tf.AUTO_REUSE):
        # Flatten the input tensor if it's not already flat
        input_shape = inputs.get_shape().as_list()
        if len(input_shape) > 2:
            inputs = tf.layers.flatten(inputs)

        # Create fully connected layer
        fc = tf.layers.dense(inputs, units=num_units, activation=None)

        # Apply batch normalization (if bn_decay and is_training are provided)
        if bn_decay is not None:
            fc = batch_norm_for_fc(fc, is_training, bn_decay, scope='bn')

        # Apply activation function
        if activation_fn is not None:
            fc = activation_fn(fc)

        return fc

'''三维最大池化'''
'''data_format='NDHWC'''
def max_pooling_3D(inputs, 
                   kernel_size, 
                   strides, 
                   padding = 'VALID'):
	
	kernel_d, kernel_h, kernel_w = kernel_size
	ksize = [1, kernel_d, kernel_h, kernel_w, 1]
	
	stride_d, stride_h, stride_w = strides
	stride = [1, stride_d, stride_h, stride_w, 1]
	
	result = tf.nn.max_pool3d(inputs, ksize, stride, padding)
	
	return result

'''三维平均池化'''
'''data_format='NDHWC'''
def avg_pooling_3D(inputs, 
                   kernel_size, 
                   strides, 
                   padding = 'VALID'):
	
	kernel_d, kernel_h, kernel_w = kernel_size
	ksize = [1, kernel_d, kernel_h, kernel_w, 1]
	
	stride_d, stride_h, stride_w = strides
	stride = [1, stride_d, stride_h, stride_w, 1]
	
	result = tf.nn.avg_pool3d(inputs, ksize, stride, padding)
	
	return result
	
'''三维反卷积'''
def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
	
	dim_size *= stride_size
	
	if padding == 'VALID' and dim_size is not None:
		dim_size += max(kernel_size - stride_size, 0)
		
	return dim_size
          
def transport_conv3D(inputs,
                     kernel_size, input_channels, output_channels,
                     strides,
                     padding = 'SAME',
                     seed = 0,
                     bn = False,
					 bn_decay = None,
					 is_training = None):
	
	kernel_d, kernel_h, kernel_w = kernel_size
	kernel_shape = [kernel_d, kernel_h, kernel_w, output_channels, input_channels]
	w = weight_variable(kernel_shape, seed)
	
	b = bias_variable([output_channels])
	
	stride_d, stride_h, stride_w = strides
	stride = [1, stride_d, stride_h, stride_w, 1]
	
	#output shape
	batch_size = inputs.get_shape()[0].value
	deep = inputs.get_shape()[1].value
	height = inputs.get_shape()[2].value
	width = inputs.get_shape()[3].value
	out_deep = get_deconv_dim(deep, stride_d, kernel_d, padding)
	out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
	out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
	output_shape = [batch_size, out_deep, out_height, out_width, output_channels]
    
	result = tf.nn.bias_add(tf.nn.conv3d_transpose(inputs, w, output_shape, stride, padding = "SAME"), b)
	
	result = tf.nn.relu(result)
	if bn:
		result = batch_norm_for_conv3d(result, is_training, bn_decay=bn_decay, scope='bn')
	return result

'''卷积模型'''  ###########################################################这里设置一个新的模型
def get_model(inputs, is_training, bn_decay = None):     #模型参数
	
	'''
	加深数据的特征通道，
	根据接收域的大小可以用3个3×3×3卷积替换一个7×7×7卷积，
	同理，2个3×3×3卷积替换一个5×5×5卷积，
	该操作可以减少计算量，减少计算参数。
	'''
	#input:16×16×16×11
	layer = conv3D(inputs, [3,3,3], 7, 20, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	#input:16×16×16×20
	layer = conv3D(layer, [3,3,3], 20, 32, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	#input:16×16×16×32
	layer = conv3D(layer, [3,3,3], 32, 64, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	#input:16×16×16×64, output:8×8×8×64
	layer_one_out = max_pooling_3D(layer, [3,3,3], [2,2,2], padding = 'SAME')
	
	'''
	valid: no padding
	same: padding with 0
	'''
	
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	#input:8×8×8×64
	layer = conv3D(layer_one_out, [3,3,3], 64, 128, [1,1,1],
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	#input:8×8×8×128
	layer = conv3D(layer, [3,3,3], 128, 192, [1,1,1],
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	#input:8×8×8×192, output:4×4×4×192
	layer = max_pooling_3D(layer, [3,3,3], [2,2,2], padding = 'SAME')
	
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	#input:4×4×4×192; output:4×4×4×64
	layer_1 = conv3D(layer, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×128
	layer_2 = conv3D(layer, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 32, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 64, 128, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×32
	layer_3 = conv3D(layer, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_3 = conv3D(layer_3, [3,3,3], 32, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×32
	layer_4 = max_pooling_3D(layer, [3,3,3], [1,1,1], padding = 'SAME')
	                 
	layer_4 = conv3D(layer_4, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#output:4×4×4×256
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)
	'''
	layer = conv3D(layer, [1,1,1], 256, 256, [1,1,1],
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	layer = max_pooling_3D(layer, [3,3,3], [2,2,2], padding = 'VALID')
	'''
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	#input:4×4×4×256; output:4×4×4×64
	layer_1 = conv3D(layer, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×128
	layer_2 = conv3D(layer, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 64, 128, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 128, 256, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×32
	layer_3 = conv3D(layer, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_3 = conv3D(layer_3, [3,3,3], 64, 128, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×32
	layer_4 = max_pooling_3D(layer, [3,3,3], [1,1,1], padding = 'SAME')
	                 
	layer_4 = conv3D(layer_4, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#output:4×4×4×512
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	layer = avg_pooling_3D(layer, [3,3,3], [1,1,1], padding = 'SAME')
	
	#upsampling:input:8×4×4×512; output:16×8×8×512
	layer = transport_conv3D(layer, [3,3,3], 512, 512, [2,2,2],
							 padding = 'SAME', seed = 0, bn = True,
							 bn_decay = bn_decay, is_training = is_training)
						 
	#output: 8×8×8×576
	layer = tf.concat([layer, layer_one_out], -1)
	
	#output: 16×16×16×576
	layer = transport_conv3D(layer, [3,3,3], 576, 576, [2,2,2],
							 padding = 'SAME', seed = 0, bn = True,
							 bn_decay = bn_decay, is_training = is_training)
	
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	#减少特征通道
	layer = conv3D(layer, [3,3,3], 576, 256, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	layer = conv3D(layer, [1,1,1], 256, 128, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	layer = conv3D(layer, [3,3,3], 128, 64, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	layer = conv3D(layer, [1,1,1], 64, 32, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	layer = conv3D(layer, [3,3,3], 32, 16, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	layer = conv3D(layer, [1,1,1], 16, 2, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	#output:16×16×16×2 因为是二分类，所以输出两个特征图
	
	return layer


def get_model1(inputs, is_training, bn_decay=None):  # 模型参数

	'''
	加深数据的特征通道，
	根据接收域的大小可以用3个3×3×3卷积替换一个7×7×7卷积，
	同理，2个3×3×3卷积替换一个5×5×5卷积，
	该操作可以减少计算量，减少计算参数。
	'''
	# input:16×16×16×11
	t1=time.time()
	layer = conv3D(inputs, [7, 7, 7], 7, 64, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)
		# input:16×16×16×64, output:8×8×8×64
	layer_one_out = max_pooling_3D(layer, [3, 3, 3], [2, 2, 2], padding='SAME')

	'''
	valid: no padding
	same: padding with 0
	'''

	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	# input:8×8×8×64
	layer = conv3D(layer_one_out, [5, 5, 5], 64, 192, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)

	# input:8×8×8×192, output:4×4×4×192
	layer = max_pooling_3D(layer, [3, 3, 3], [2, 2, 2], padding='SAME')
	print("大卷积核运行时间：", time.time() - t1)
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	# input:4×4×4×192; output:4×4×4×64
	layer_1 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×128
	layer_2 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 32, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×32
	layer_3 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_3 = conv3D(layer_3, [3, 3, 3], 32, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×32
	layer_4 = max_pooling_3D(layer, [3, 3, 3], [1, 1, 1], padding='SAME')

	layer_4 = conv3D(layer_4, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# output:4×4×4×256
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)

	# input:4×4×4×256; output:4×4×4×64
	layer_1 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×128
	layer_2 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 128, 256, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×32
	layer_3 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_3 = conv3D(layer_3, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×32
	layer_4 = max_pooling_3D(layer, [3, 3, 3], [1, 1, 1], padding='SAME')

	layer_4 = conv3D(layer_4, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# output:4×4×4×512
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)

	return layer


def get_modeln(inputs, is_training, bn_decay=None):  # 模型参数

	'''
	加深数据的特征通道，
	根据接收域的大小可以用3个3×3×3卷积替换一个7×7×7卷积，
	同理，2个3×3×3卷积替换一个5×5×5卷积，
	该操作可以减少计算量，减少计算参数。
	'''
	# input:16×16×16×11
	layer = conv3D(inputs, [7, 7, 7], 7, 64, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)


	# input:16×16×16×64, output:8×8×8×64
	layer_one_out = max_pooling_3D(layer, [3, 3, 3], [2, 2, 2], padding='SAME')

	'''
	valid: no padding
	same: padding with 0
	'''

	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	# input:8×8×8×64
	layer = conv3D(layer_one_out, [5, 5, 5], 64, 192, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)

	# input:8×8×8×192, output:4×4×4×192
	layer = max_pooling_3D(layer, [3, 3, 3], [2, 2, 2], padding='SAME')

	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	# input:4×4×4×192; output:4×4×4×64
	layer_1 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×128
	layer_2 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 32, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×32
	layer_3 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_3 = conv3D(layer_3, [3, 3, 3], 32, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×32
	layer_4 = max_pooling_3D(layer, [3, 3, 3], [1, 1, 1], padding='SAME')

	layer_4 = conv3D(layer_4, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# output:4×4×4×256
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)
	'''
	layer = conv3D(layer, [1,1,1], 256, 256, [1,1,1],
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)

	layer = max_pooling_3D(layer, [3,3,3], [2,2,2], padding = 'VALID')
	'''
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	# input:4×4×4×256; output:4×4×4×64
	layer_1 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×128
	layer_2 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 128, 256, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×32
	layer_3 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_3 = conv3D(layer_3, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4×4×4×192; output:4×4×4×32
	layer_4 = max_pooling_3D(layer, [3, 3, 3], [1, 1, 1], padding='SAME')

	layer_4 = conv3D(layer_4, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# output:4×4×4×512
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	layer = avg_pooling_3D(layer, [3, 3, 3], [1, 1, 1], padding='SAME')

	# upsampling:input:8×4×4×512; output:16×8×8×512
	layer = transport_conv3D(layer, [3, 3, 3], 512, 512, [2, 2, 2],
							 padding='SAME', seed=0, bn=True,
							 bn_decay=bn_decay, is_training=is_training)

	# output: 8×8×8×576
	layer = tf.concat([layer, layer_one_out], -1)

	# output: 16×16×16×576
	layer = transport_conv3D(layer, [3, 3, 3], 576, 576, [2, 2, 2],
							 padding='SAME', seed=0, bn=True,
							 bn_decay=bn_decay, is_training=is_training)

	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	# 减少特征通道
	layer = conv3D(layer, [3, 3, 3], 576, 256, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)

	layer = conv3D(layer, [1, 1, 1], 256, 128, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)

	layer = conv3D(layer, [3, 3, 3], 128, 64, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)

	layer = conv3D(layer, [1, 1, 1], 64, 32, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)

	layer = conv3D(layer, [3, 3, 3], 32, 16, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)

	layer = conv3D(layer, [1, 1, 1], 16, 2, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)
	# output:16×16×16×2 因为是二分类，所以输出两个特征图

	return layer


'''
dice loss
考虑两个类别
'''
#徐章皓修改内容
# def dice_loss(output, target, loss_type='jaccard', axis = (0,1), smooth=1e-5):
def dice_loss(output, target, loss_type='jaccard', axis=(1,2,3,4), smooth=1e-5):
	"""
	Soft dice (Sorensen or Jaccard) coefficient for comparing the similarity of two batch of data, 
	usually be used for binary image segmentation
	i.e. labels are binary. 
	The coefficient between 0 to 1, 1 means totally match.
	
	Parameters
	-----------
	output : Tensorshape: [batch_size
		A distribution with , ....], (any dimensions).
	target : Tensor
		The target distribution, format the same with `output`.
	loss_type : str
		``jaccard`` or ``sorensen``, default is ``jaccard``.
	axis : tuple of int
		All dimensions are reduced, default ``[1,2,3]``.
	smooth : float
	This small value will be added to the numerator and denominator.
		- If both output and target are empty, it makes sure dice is 1.
		- If either output or target are empty (all pixels are background), 
		dice = '''smooth/(small_value + smooth)''', 
		then if smooth is very small, 
		dice close to 0 (even the image values lower than the threshold), 
		so in this case, higher smooth can have a higher dice.
		
	Examples
	---------
		>>> outputs = tl.act.pixel_wise_softmax(network.outputs)
		>>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
	References
	-----------
		- `Wiki-Dice <https://en.wikipedia.org/wiki/Sorensen–Dice_coefficient>`__
	"""
	
	#将非稀疏标签转换为稀疏标签
	#tf.one_hot(indices, depth, on_value, off_value, axis)
	#  indices是一个列表,指定张量中独热向量的独热位置，或者说indeces是非负整数表示的标签列表。len(indices)就是分类的类别数。
	#  tf.one_hot返回的张量的阶数为indeces的阶数+1。
	#  当indices的某个分量取-1时，即对应的向量没有独热值。
	#  depth是每个独热向量的维度
	#  on_value是独热值
	#  off_value是非独热值
	#  axis指定第几阶为depth维独热向量，默认为-1,即,指定张量的最后一维为独热向量
	#  例如：对于一个2阶张量而言,axis=0时，即，每个列向量是一个独热的depth维向量
	#  axis=1时，即，每个行向量是一个独热的depth维向量。axis=-1，等价于axis=1
	#  tf.one_hot(indices, depth, on_value, off_value, axis)
	'''
	predict = tf.nn.softmax(predict)
	
	gt = tf.to_float(gt)
	gt = tf.reshape(gt, [-1])
	
	predict = tf.reshape(predict, [-1])
	shape = predict.get_shape()[0].value
	
	location = list(range(1,shape,2))
	indices = []
	for i in range(len(location)):
		a = []
		a.append(location[i])
		indices.append(a)
	predict = tf.gather_nd(predict,indices)
	
	intersection = tf.reduce_sum(predict * gt, axis=axis)
	union = tf.reduce_sum(predict * predict, axis=axis) + tf.reduce_sum(gt * gt, axis=axis)
	dice_coefficient = tf.reduce_mean(2 * (intersection + smooth)/(union + smooth))
	dice = 1 - dice_coefficient
	'''

	# N = len(np.array(target).reshape(-1))  师姐原来内容
	N = tf.reduce_prod(tf.shape(target))
	# target = tf.to_float(tf.one_hot(indices = target, depth = 2, on_value = 1, off_value = 0))
	target = tf.cast(tf.one_hot(indices = target, depth = 2, on_value = 1, off_value = 0),dtype = tf.float32)#ORG:(71,4096,2)

	# 徐章皓修改
	output = tf.nn.softmax(output)#ORG:(71,4096,2)

	print(output*target)
	#dice loss只能是稀疏类型标签，独热编码
	inse = tf.reduce_sum(output * target, axis=axis)#ORG:(71,)sum(4096) FC:(71,)sum(2)
	
	if loss_type == 'jaccard':
		l = tf.reduce_sum(output * output, axis=axis)
		r = tf.reduce_sum(target * target, axis=axis)
	elif loss_type == 'sorensen':
		l = tf.reduce_sum(output, axis=axis)
		r = tf.reduce_sum(target, axis=axis)
	else:
		raise Exception("Unknow loss_type")
		
	dice_coe = (2. * inse + smooth) / (l + r + smooth)
	N = tf.cast(N,tf.float32)
	dice_coe = tf.reduce_mean(dice_coe) / N
	# dice_coe = tf.reduce_mean(dice_coe)
	dice = 1 - dice_coe

	return dice

'''
focal loss
'''
def focal_loss(pred, label, alpha = 0.25, gamma = 2):
	
	alpha = tf.constant(alpha, dtype=tf.float32)
	gamma = tf.constant(gamma, dtype=tf.float32)
	epsilon = 1.e-8
	
	y_true = tf.one_hot(label, 2) #二分类ORG:(71,4096,2)
	probs = tf.nn.sigmoid(pred) #用的是sigmoid(71,4096,2)
	
	y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)#(71,4096,2)
	
	weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), gamma))#(71,4096,2)
	if alpha != 0.0:
		alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)#(71,4096,2)
	else:
		alpha_t = tf.ones_like(y_true)
		
	xent = tf.multiply(y_true, -tf.log(y_pred))
	focal_xent = tf.multiply(alpha_t, tf.multiply(weight, xent))
	reduced_fl = tf.reduce_max(focal_xent, axis=1)#(71,16,16,2)
	focal_loss = tf.reduce_mean(reduced_fl)#(16*16)
	# focal_loss = tf.reduce_mean(focal_xent)#(16*16)
	'''
	#计算调节因子weight和alpha
	# 先得到y_true和1-y_true的概率，正负样本都进行计算
	p_t = y_true*y_pred+(tf.ones_like(y_true)-y_true)*(tf.ones_like(y_true)-y_pred)
	
	#通过p_t和gamma得到weight
	weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)
	
	#得到alpha，y_true的是alpha，那么1-y_true的是1-alpha
	alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
	
	#最后就是论文中的公式，相当于：- alpha * (1-p_t)^gamma * log(p_t)
	focal_loss = - alpha_t * weight * tf.log(p_t)
	'''
	return focal_loss
	
'''
损失函数(交叉熵损失函数)
利用loss调整下降梯度、权重及偏移量

tf.nn.sparse_softmax_cross_entropy_with_logits()输入的label格式为一维向量;
tf.nn.softmax_cross_entropy_with_logits()输入的label格式为one_hot格式。

上述两个函数运行过程可以拆解为两步:
1) softmax函数将每个类别对应的输出分量归一化，使各个分类的和为1，可以理解为将输出分量
   转换为input data分类为每个类别的概率。
2) Cross_Entropy作为loss function计算loss。
'''
def get_loss(pred, label):   ######似乎没用混合损失函数，只用了交叉熵
	
	# seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label) #非稀疏
	# seg = tf.reduce_mean(seg_loss)
	# hybird_loss = seg
	dice = dice_loss(pred, label,axis=1)
	focal = focal_loss(pred, label, alpha = 0.8, gamma = 2)
	hybird_loss = dice+focal

	return hybird_loss
