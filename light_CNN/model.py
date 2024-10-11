# encoding:gbk
from __future__ import division
import os
import csv
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.model_selection import KFold
import time


'''��֯���ݸ�ʽ'''
def change_format(data, batchsize, deep, height, weight, channel, mark):
	"""
	   ��һά����ת��Ϊָ��ά�ȵĶ�ά�����������

	   Parameters:
	   - data: �����һά����
	   - batchsize: ���δ�С
	   - deep: �������
	   - height: ���ݸ߶�
	   - weight: ���ݿ��
	   - channel: ����ͨ����
	   - mark: 'TRUE' ��ʾת��Ϊ��ά���飬'FALSE' ��ʾת��Ϊ��ά����

	   Returns:
	   - outdata: ת����Ķ�ά���������
	   """
	data = np.array(data)
	if mark == 'TRUE':
		outdata = data.reshape(batchsize, deep, height, weight, channel)
		#��һά������nά����
		#����ת���飬�б���ת������������������������
		
	elif mark == 'FALSE':
		outdata = data.reshape(batchsize, deep, height, weight)
		#��һά������nά����
	
	return outdata

'''��ȡcsv����'''
'''
����Ԫ���ζ�ȡ�������ɣ�NDWHC�����������ڵĸ�����
eg.[a1,a2,a3,b1,b2,b3,c1,c2,c3]
N:batch_size; D:K; W:J; H:I; C:Feature
'''
def read_csv(filepath, filenames, batchsize, deep, height, weight, channel):
	"""
	    ��CSV�ļ��ж�ȡ���ݲ�����Ԥ����

	    Parameters
	    ----------
	    filepath : str
	        CSV�ļ����ڵ�Ŀ¼·����
	    filenames : list of str
	        ����CSV�ļ������б�
	    batchsize : int
	        ÿ��batch������������
	    deep : int
	        ���ݵ���ȣ�����ά������ͨ����ʾZ��Ĵ�С����
	    height : int
	        ���ݵĸ߶ȣ�����ά������ͨ����ʾY��Ĵ�С����
	    weight : int
	        ���ݵĿ�ȣ�����ά������ͨ����ʾX��Ĵ�С����
	    channel : int
	        ���ݵ�ͨ������

	    Returns
	    -------
	    outdata : list
	        ���������ݣ����б���ʽ���ء�
	    label : list
	        �����ı�ǩ�����б���ʽ���ء�

	    """
	data = []
	label = []
	
	#open()��ȡ����·�����ļ��������ݣ������Ҫ��·����Ϊ�����������һ��ƴ��
	# for name in filenames:
	#
	# 	csvfile = open(os.path.join(filepath, name), 'r')
	# 	reader = csv.reader(csvfile)
	# 	for item in reader:
	# 		if reader.line_num == 1: #��һ��Ϊ������
	# 			continue
	# 		#data.append(item[3:-2]) #ǰ����ΪI,J,K; ������Ϊ��ǩ
	# 		#[3:-2]��ָ�ӵ�4�е�����������
	# 		a = np.array(item[3:10])
	# 		#b = np.array(item[16:20])
	# 		#data.append(np.concatenate((a,b)).tolist())
	# 		data.append(a.tolist())
	# 		label.append(item[-1]) #������Ϊ��ǩ����ȡ���һ��Ϊ��ϡ��
	#
	# data = np.array(data)
	# label = np.array(label)
	# data = data.flatten() #����ά����ת����һά����
	# data = data.astype(np.float) #���ַ���ת��Ϊfloat
	# label = label.astype(np.int)
	# data = data.tolist() #����ת�б�
	# label = label.tolist()
	# outdata = change_format(data, batchsize, deep, height, weight, channel, 'TRUE') #�б�ת��ά����
	# label = change_format(label, batchsize, deep, height, weight, channel, 'FALSE') #�б�ת��ά���飬��ϡ��
	#
	# return outdata, label
	for name in filenames:

		csvfile = open(os.path.join(filepath, name), 'r') 							  # ��CSV�ļ�
		reader = csv.reader(csvfile)												  # ʹ��CSV��ȡ����ȡ�ļ�����
		for item in reader:															  # ����CSV�ļ���ÿһ��
			if reader.line_num == 1:
				continue															  # ������һ�У���Ϊ��ͨ������������
			# data.append(item[3:-2]) #ǰ����ΪI,J,K; ������Ϊ��ǩ
			# [3:-2]��ָ�ӵ�4�е�����������
			a = np.array(item[3:10])
			# b = np.array(item[16:20])
			# data.append(np.concatenate((a,b)).tolist())
			data.append(a.tolist()) 												  # ������ת��Ϊ�б���ӵ�data�б���
			label.append(item[-1])  												  # ������Ϊ��ǩ����ȡ���һ��Ϊ��ϡ��
	# �����ݺͱ�ǩת��ΪNumPy����
	data = np.array(data)
	label = np.array(label)
	data = data.flatten()  															  # ����ά����ת����һά����
	data = data.astype(np.float) 													  # ���ַ���ת��Ϊfloat
	label = label.astype(np.int)
	data = data.tolist()  															  # ����ת�б�
	label = label.tolist()
	outdata = change_format(data, batchsize, deep, height, weight, channel, 'TRUE')   # �б�ת��ά����
	label = change_format(label, batchsize, deep, height, weight, channel, 'FALSE')   # �б�ת��ά���飬��ϡ��

	return outdata, label
'''k�۽�����֤'''
def cross_validation(filelist, fold):
 	'''
    Input:
    filelist: �ļ����б���������ʽ�ṩ
    fold: ��������������֤������

    Output:
    train_list: ����ÿ�����ѵ�����ļ������б�
    test_list: ����ÿ�������Լ��ļ������б�

    ʹ��KFold�������н�����֤�������ݰ���ָ����������Ϊѵ�����Ͳ��Լ���
	'''

	filelist = np.array(filelist)
	
	kf = KFold(n_splits = fold, random_state=2020, shuffle=True)
	#random_state: ��������ӣ�ֻ����shuffleΪTRUEʱ��Ч
	kf.get_n_splits(filelist) # ��ѯ�ֳɼ�����
	
	train_list = []
	test_list = []
	
	#���ֺ󴫳���������,������ת��Ϊ�����б�
	for train_index, test_index in kf.split(filelist):
		X_train,X_test = filelist[train_index],filelist[test_index]
		train_list.append(X_train.tolist())
		test_list.append(X_test.tolist())
	
	return train_list, test_list
	
'''
��ʮ�۽�����֤�Ľ���ļ����洢��txt��
'''
	
'''ռλ��'''
def placeholder_inputs(batch_size, size, channel):
	"""
	   ���������������ݵ� TensorFlow ռλ����

	   Parameters:
	   - batch_size: ���δ�С
	   - size: ���ݵĳߴ磨������ά���ϵĴ�С��ͬ���������������ݣ�
	   - channel: ����ͨ����

	   Returns:
	   - pointclouds_pl: ����������ݵ�ռλ������״Ϊ (batch_size, size, size, size, channel)
	   - labels_pl: �����ǩ���ݵ�ռλ������״Ϊ (batch_size, size, size, size)
	   """
	# ������������������ݵ�ռλ������������Ϊ float32
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, size, size, size, channel))
	# �������������ǩ���ݵ�ռλ������������Ϊ int32����ϡ���ʾ
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, size, size, size)) #��ϡ��
    
    return pointclouds_pl, labels_pl

'''���'''
def log_string(LOG_FOUT, out_str):
	"""
	    ����ַ�������־�ļ��Ϳ���̨

	    Parameters:
	    - LOG_FOUT: �ļ��������ʾ��־�ļ�
	    - out_str: Ҫ������ַ���
	    """
    LOG_FOUT.write(out_str+'\n')												 # ���ַ���д�뵽��־�ļ���ĩβ��ӻ��з�
    LOG_FOUT.flush()															 # ����ˢ����־�ļ���ȷ��д�����ݼ�ʱ��Ч
    print(out_str)  															 # �ڿ���̨��ӡ��ͬ���ַ���
    
'''����ѧϰ��'''
def get_learning_rate(base_learning_rate, 
                      batch, batch_size,
                      decay_step,
                      decay_rate):
	"""
        ��ȡ��̬������ѧϰ�ʡ�

        Parameters:
        - base_learning_rate: ��ʼѧϰ��
        - batch: ������
        - batch_size: ���δ�С
        - decay_step: ѧϰ��˥������
        - decay_rate: ѧϰ��˥����

        Returns:
        - learning_rate: �������ѧϰ��
        """
	# ʹ��ָ��˥����������ѧϰ�ʵĶ�̬����
    learning_rate = tf.train.exponential_decay(
                        base_learning_rate,  							# ��ʼѧϰ��
                        batch * batch_size,  							# ��ǰ�������������γ������δ�С��
                        decay_step,          							# ѧϰ��˥������
                        decay_rate,         							# ѧϰ��˥����
                        staircase=True)	 								# �Ƿ�ʹ�ý���ʽ��Staircase��˥��
    learning_rate = tf.maximum(learning_rate, 0.00001) 					# ��ѧϰ��������һ����Сֵ��0.00001�����ϣ�����ѧϰ�ʹ�С
    
    return learning_rate        

'''����˥����'''
"""
    ��ȡ��̬����������һ��˥��ϵ����

    Parameters:
    - bn_init_decay: ��ʼ����һ��˥��ϵ��
    - batch: ��ǰ������
    - batch_size: ���δ�С
    - bn_decay_decay_step: ����һ��˥������
    - bn_decay_decay_rate: ����һ��˥����
    - bn_decay_clip: ����һ��˥��ϵ��������ֵ

    Returns:
    - bn_decay: �����������һ��˥��ϵ��
    """
# ʹ��ָ��˥��������������һ��˥��ϵ���Ķ�̬����
def get_bn_decay(bn_init_decay,											# ��ʼ����һ��˥��ϵ��
                 batch, batch_size, 									# ��ǰ�������������γ������δ�С��
                 bn_decay_decay_step,									# ����һ��˥������
                 bn_decay_decay_rate, 									# ����һ��˥����
                 bn_decay_clip):
    bn_momentum = tf.train.exponential_decay(
                      bn_init_decay,
                      batch * batch_size,
                      bn_decay_decay_step,
                      bn_decay_decay_rate,
                      staircase=True)
    bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)				# ������һ��˥��ϵ��������һ�����ֵ��bn_decay_clip������
    
    return bn_decay

'''
����������һ��:
ʵ���������������м����Ԥ����Ĳ�����������һ��������һ��������ٽ����������һ�㣬
��������Ч�ط�ֹ���ݶ���ɢ������������ѵ�����ӿ�������
���Լ�������ȡ��DropOut��ʹ�á�
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
  
'''��ά�����һ��'''
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

'''ȫ���ӹ�һ��'''
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

'''y=wx+b, wΪweight_variable, bΪbias_variable'''
'''��ʼ��������, w������ͼ�����'''
def weight_variable(shape, seed=0):
	
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed) #��ʼ����ֵ,������̬�ֲ�,��׼��Ϊ0.1
    w = tf.Variable(initial)
    
    return w
 
'''��ʼ��ƫ��, b�����ͼ�����'''
def bias_variable(shape):
	
    initial = tf.constant(0.1, shape=shape) #��ʼ��bias,��0.1���
    b = tf.Variable(initial)
    
    return b
    
'''��ά���'''
'''
data_format����ѡ��string,�����ǣ�"NDHWC", "NCDHW"��Ĭ��Ϊ"NDHWC".�����������ݵ����ݸ�ʽ��
ʹ��Ĭ�ϸ�ʽ"NDHWC",���ݰ�����˳��洢��[batch,in_depth,in_height,in_width,in_channels]��
����,��ʽ�����ǡ�NCDHW��,���ݴ洢˳���ǣ�[batch,in_channels,in_depth,in_height,in_width]��
�������ݵ�����Ϊdtype = tf.float32,������ܻ����
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
	"""
	   3D ���������

	   ������
	   - inputs: ����������
	   - kernel_size: ����˴�С����ʽΪ [depth, height, width]��
	   - input_channels: ����ͨ������
	   - output_channels: ���ͨ������
	   - strides: ��������ʽΪ [depth_stride, height_stride, width_stride]��
	   - padding: ��䷽ʽ��Ĭ��Ϊ 'SAME'��
	   - seed: ������ӣ�����Ȩ�س�ʼ����
	   - dp: �Ƿ�ʹ�� dropout��
	   - bn: �Ƿ�ʹ������һ����
	   - bn_decay: ����һ����˥��ϵ����
	   - is_training: �Ƿ���ѵ���׶Ρ�

	   ���أ�
	   - result: ��������������ѡ�� dropout ������һ��������������
	   """
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
'''��άȫ����'''
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

'''��ά���ػ�'''
'''data_format='NDHWC'''
def max_pooling_3D(inputs, 
                   kernel_size, 
                   strides, 
                   padding = 'VALID'):
	"""
	    3D ���ػ�������

	    ������
	    - inputs: ����������
	    - kernel_size: �ػ��˴�С����ʽΪ [depth, height, width]��
	    - strides: ��������ʽΪ [depth_stride, height_stride, width_stride]��
	    - padding: ��䷽ʽ��Ĭ��Ϊ 'VALID'��

	    ���أ�
	    - result: �������ػ��������������
	    """
	kernel_d, kernel_h, kernel_w = kernel_size
	ksize = [1, kernel_d, kernel_h, kernel_w, 1]
	
	stride_d, stride_h, stride_w = strides
	stride = [1, stride_d, stride_h, stride_w, 1]
	
	result = tf.nn.max_pool3d(inputs, ksize, stride, padding)
	
	return result

'''��άƽ���ػ�'''
'''data_format='NDHWC'''
def avg_pooling_3D(inputs,
                   kernel_size,
                   strides,
                   padding = 'VALID'):
	"""
	    3D ƽ���ػ�������

	    ������
	    - inputs: ����������
	    - kernel_size: �ػ��˴�С����ʽΪ [depth, height, width]��
	    - strides: ��������ʽΪ [depth_stride, height_stride, width_stride]��
	    - padding: ��䷽ʽ��Ĭ��Ϊ 'VALID'��

	    ���أ�
	    - result: ����ƽ���ػ��������������
	    """
	kernel_d, kernel_h, kernel_w = kernel_size
	ksize = [1, kernel_d, kernel_h, kernel_w, 1]

	stride_d, stride_h, stride_w = strides
	stride = [1, stride_d, stride_h, stride_w, 1]

	result = tf.nn.avg_pool3d(inputs, ksize, stride, padding)

	return result

'''��ά�����'''
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
	"""
	   3D ת�þ�����������������

	   ������
	   - inputs: ����������
	   - kernel_size: ����˴�С����ʽΪ [depth, height, width]��
	   - input_channels: ����ͨ������
	   - output_channels: ���ͨ������
	   - strides: ��������ʽΪ [depth_stride, height_stride, width_stride]��
	   - padding: ��䷽ʽ��Ĭ��Ϊ 'SAME'��
	   - seed: ������ӡ�
	   - bn: �Ƿ�ʹ������һ����
	   - bn_decay: ����һ��˥��������
	   - is_training: �Ƿ���ѵ��ģʽ��

	   ���أ�
	   - result: ����ת�þ���������������
	   """
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

def get_model(inputs, is_training, bn_decay = None):
	
	'''
	�������ݵ�����ͨ����
	���ݽ�����Ĵ�С������3��3��3��3����滻һ��7��7��7�����
	ͬ��2��3��3��3����滻һ��5��5��5�����
	�ò������Լ��ټ����������ټ��������
	'''
	#input:16��16��16��7
	layer = conv3D(inputs, [3,3,3], 7, 20, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	#input:16��16��16��20
	layer = conv3D(layer, [3,3,3], 20, 32, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	#input:16��16��16��32
	layer = conv3D(layer, [3,3,3], 32, 64, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	#input:16��16��16��64, output:8��8��8��64
	layer_one_out = max_pooling_3D(layer, [3,3,3], [2,2,2], padding = 'SAME')
	
	'''
	valid: no padding
	same: padding with 0
	'''
	
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	#input:8��8��8��64
	layer = conv3D(layer_one_out, [3,3,3], 64, 128, [1,1,1],
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	#input:8��8��8��128
	layer = conv3D(layer, [3,3,3], 128, 192, [1,1,1],
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	#input:8��8��8��192, output:4��4��4��192
	layer = max_pooling_3D(layer, [3,3,3], [2,2,2], padding = 'SAME')
	
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	#inceptionģ��
	#input:4��4��4��192; output:4��4��4��64
	layer_1 = conv3D(layer, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4��4��4��192; output:4��4��4��128
	layer_2 = conv3D(layer, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 32, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 64, 128, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4��4��4��192; output:4��4��4��32
	layer_3 = conv3D(layer, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_3 = conv3D(layer_3, [3,3,3], 32, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4��4��4��192; output:4��4��4��32
	layer_4 = max_pooling_3D(layer, [3,3,3], [1,1,1], padding = 'SAME')
	                 
	layer_4 = conv3D(layer_4, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#output:4��4��4��256
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)

	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	#inception
	#input:4��4��4��256; output:4��4��4��64
	layer_1 = conv3D(layer, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	#input:4��4��4��192; output:4��4��4��128
	layer_2 = conv3D(layer, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 64, 128, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 128, 256, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4��4��4��192; output:4��4��4��32
	layer_3 = conv3D(layer, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_3 = conv3D(layer_3, [3,3,3], 64, 128, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4��4��4��192; output:4��4��4��32
	layer_4 = max_pooling_3D(layer, [3,3,3], [1,1,1], padding = 'SAME')
	                 
	layer_4 = conv3D(layer_4, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#output:4��4��4��512
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	layer = avg_pooling_3D(layer, [3,3,3], [1,1,1], padding = 'SAME')
	
	#upsampling:input:8��4��4��512; output:16��8��8��512
	layer = transport_conv3D(layer, [3,3,3], 512, 512, [2,2,2],
							 padding = 'SAME', seed = 0, bn = True,
							 bn_decay = bn_decay, is_training = is_training)
						 
	#output: 8��8��8��576
	layer = tf.concat([layer, layer_one_out], -1)
	
	#output: 16��16��16��576
	layer = transport_conv3D(layer, [3,3,3], 576, 576, [2,2,2],
							 padding = 'SAME', seed = 0, bn = True,
							 bn_decay = bn_decay, is_training = is_training)
	
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	#��������ͨ��
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
	#output:16��16��16��2 ��Ϊ�Ƕ����࣬���������������ͼ
	
	return layer


def get_model1(inputs, is_training, bn_decay=None):  # ģ�Ͳ���

	'''
	�������ݵ�����ͨ����
	���ݽ�����Ĵ�С������3��3��3��3����滻һ��7��7��7�����
	ͬ��2��3��3��3����滻һ��5��5��5�����
	�ò������Լ��ټ����������ټ��������
	'''
	# input:16��16��16��11
	t1=time.time()
	layer = conv3D(inputs, [7, 7, 7], 7, 64, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)
		# input:16��16��16��64, output:8��8��8��64
	layer_one_out = max_pooling_3D(layer, [3, 3, 3], [2, 2, 2], padding='SAME')

	'''
	valid: no padding
	same: padding with 0
	'''

	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	# input:8��8��8��64
	layer = conv3D(layer_one_out, [5, 5, 5], 64, 192, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)

	# input:8��8��8��192, output:4��4��4��192
	layer = max_pooling_3D(layer, [3, 3, 3], [2, 2, 2], padding='SAME')
	print("����������ʱ�䣺", time.time() - t1)
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	# input:4��4��4��192; output:4��4��4��64
	layer_1 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��128
	layer_2 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 32, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��32
	layer_3 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_3 = conv3D(layer_3, [3, 3, 3], 32, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��32
	layer_4 = max_pooling_3D(layer, [3, 3, 3], [1, 1, 1], padding='SAME')

	layer_4 = conv3D(layer_4, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# output:4��4��4��256
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)

	# input:4��4��4��256; output:4��4��4��64
	layer_1 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��128
	layer_2 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 128, 256, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��32
	layer_3 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_3 = conv3D(layer_3, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��32
	layer_4 = max_pooling_3D(layer, [3, 3, 3], [1, 1, 1], padding='SAME')

	layer_4 = conv3D(layer_4, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# output:4��4��4��512
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)

	return layer


def get_modeln(inputs, is_training, bn_decay=None):  # ģ�Ͳ���

	'''
	�������ݵ�����ͨ����
	���ݽ�����Ĵ�С������3��3��3��3����滻һ��7��7��7�����
	ͬ��2��3��3��3����滻һ��5��5��5�����
	�ò������Լ��ټ����������ټ��������
	'''
	# input:16��16��16��11
	layer = conv3D(inputs, [7, 7, 7], 7, 64, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)


	# input:16��16��16��64, output:8��8��8��64
	layer_one_out = max_pooling_3D(layer, [3, 3, 3], [2, 2, 2], padding='SAME')

	'''
	valid: no padding
	same: padding with 0
	'''

	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	# input:8��8��8��64
	layer = conv3D(layer_one_out, [5, 5, 5], 64, 192, [1, 1, 1],
				   padding='SAME', seed=0, dp=False, bn=True,
				   bn_decay=bn_decay, is_training=is_training)

	# input:8��8��8��192, output:4��4��4��192
	layer = max_pooling_3D(layer, [3, 3, 3], [2, 2, 2], padding='SAME')

	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	# input:4��4��4��192; output:4��4��4��64
	layer_1 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��128
	layer_2 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 32, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��32
	layer_3 = conv3D(layer, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_3 = conv3D(layer_3, [3, 3, 3], 32, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��32
	layer_4 = max_pooling_3D(layer, [3, 3, 3], [1, 1, 1], padding='SAME')

	layer_4 = conv3D(layer_4, [1, 1, 1], 192, 32, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# output:4��4��4��256
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)
	'''
	layer = conv3D(layer, [1,1,1], 256, 256, [1,1,1],
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)

	layer = max_pooling_3D(layer, [3,3,3], [2,2,2], padding = 'VALID')
	'''
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	# input:4��4��4��256; output:4��4��4��64
	layer_1 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��128
	layer_2 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_2 = conv3D(layer_2, [3, 3, 3], 128, 256, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��32
	layer_3 = conv3D(layer, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	layer_3 = conv3D(layer_3, [3, 3, 3], 64, 128, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# input:4��4��4��192; output:4��4��4��32
	layer_4 = max_pooling_3D(layer, [3, 3, 3], [1, 1, 1], padding='SAME')

	layer_4 = conv3D(layer_4, [1, 1, 1], 256, 64, [1, 1, 1],
					 padding='SAME', seed=0, dp=False, bn=True,
					 bn_decay=bn_decay, is_training=is_training)

	# output:4��4��4��512
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	layer = avg_pooling_3D(layer, [3, 3, 3], [1, 1, 1], padding='SAME')

	# upsampling:input:8��4��4��512; output:16��8��8��512
	layer = transport_conv3D(layer, [3, 3, 3], 512, 512, [2, 2, 2],
							 padding='SAME', seed=0, bn=True,
							 bn_decay=bn_decay, is_training=is_training)

	# output: 8��8��8��576
	layer = tf.concat([layer, layer_one_out], -1)

	# output: 16��16��16��576
	layer = transport_conv3D(layer, [3, 3, 3], 576, 576, [2, 2, 2],
							 padding='SAME', seed=0, bn=True,
							 bn_decay=bn_decay, is_training=is_training)

	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	# ��������ͨ��
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
	# output:16��16��16��2 ��Ϊ�Ƕ����࣬���������������ͼ

	return layer


'''
dice loss
�����������
'''
#������޸�����
# def dice_loss(output, target, loss_type='jaccard', axis = (0,1), smooth=1e-5):
def dice_loss(output, target, loss_type='jaccard', axis=(1,2,3,4), smooth=1e-5):
	"""
    ���� Dice Loss������� Sorensen �� Jaccard ϵ�������ڱȽ������������ε������ԣ�ͨ�����ڶ�����ͼ��ָ�����

    ������
    - output: ���������Ԥ��������
    - target: ��ʵ��ǩ������
    - loss_type: ��ʧ���ͣ���ѡ 'jaccard' �� 'sorensen'��Ĭ��Ϊ 'jaccard'��
    - axis: ָ������ Dice Loss ʱ���м�����ᣬĬ��Ϊ (1,2,3,4)��
    - smooth: ƽ�����ֹ��ĸΪ�㣬Ĭ��Ϊ 1e-5��

    ���أ�
    - dice: Dice Loss��

		
	Examples
	---------
		>>> outputs = tl.act.pixel_wise_softmax(network.outputs)
		>>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
	References
	-----------
		- `Wiki-Dice <https://en.wikipedia.org/wiki/Sorensen�CDice_coefficient>`__
	"""
	
	#����ϡ���ǩת��Ϊϡ���ǩ
	#tf.one_hot(indices, depth, on_value, off_value, axis)
	#  indices��һ���б�,ָ�������ж��������Ķ���λ�ã�����˵indeces�ǷǸ�������ʾ�ı�ǩ�б�len(indices)���Ƿ�����������
	#  tf.one_hot���ص������Ľ���Ϊindeces�Ľ���+1��
	#  ��indices��ĳ������ȡ-1ʱ������Ӧ������û�ж���ֵ��
	#  depth��ÿ������������ά��
	#  on_value�Ƕ���ֵ
	#  off_value�ǷǶ���ֵ
	#  axisָ���ڼ���Ϊdepthά����������Ĭ��Ϊ-1,��,ָ�����������һάΪ��������
	#  ���磺����һ��2����������,axis=0ʱ������ÿ����������һ�����ȵ�depthά����
	#  axis=1ʱ������ÿ����������һ�����ȵ�depthά������axis=-1���ȼ���axis=1
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

	# N = len(np.array(target).reshape(-1))  ʦ��ԭ������
	N = tf.reduce_prod(tf.shape(target))
	# target = tf.to_float(tf.one_hot(indices = target, depth = 2, on_value = 1, off_value = 0))
	target = tf.cast(tf.one_hot(indices = target, depth = 2, on_value = 1, off_value = 0),dtype = tf.float32)#ORG:(71,4096,2)

	# ������޸�
	output = tf.nn.softmax(output)#ORG:(71,4096,2)

	print(output*target)
	#dice lossֻ����ϡ�����ͱ�ǩ�����ȱ���
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
	"""
	    ���� Focal Loss�����ڴ������ƽ�����⣬�ر������ڶ���������

	    ������
	    - pred: ���������Ԥ��������
	    - label: ��ʵ��ǩ������
	    - alpha: ƽ�����ӣ�������������Ȩ�أ�Ĭ��Ϊ 0.25��
	    - gamma: Focal Loss �ĵ������ӣ�Ĭ��Ϊ 2��

	    ���أ�
	    - focal_loss: Focal Loss ��ֵ��
	    """
	alpha = tf.constant(alpha, dtype=tf.float32)
	gamma = tf.constant(gamma, dtype=tf.float32)
	epsilon = 1.e-8
	
	y_true = tf.one_hot(label, 2) #������ORG:(71,4096,2)
	probs = tf.nn.sigmoid(pred) #�õ���sigmoid(71,4096,2)
	
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
	#�����������weight��alpha
	# �ȵõ�y_true��1-y_true�ĸ��ʣ��������������м���
	p_t = y_true*y_pred+(tf.ones_like(y_true)-y_true)*(tf.ones_like(y_true)-y_pred)
	
	#ͨ��p_t��gamma�õ�weight
	weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)
	
	#�õ�alpha��y_true����alpha����ô1-y_true����1-alpha
	alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
	
	#�����������еĹ�ʽ���൱�ڣ�- alpha * (1-p_t)^gamma * log(p_t)
	focal_loss = - alpha_t * weight * tf.log(p_t)
	'''
	return focal_loss
	
'''
��ʧ����(��������ʧ����)
����loss�����½��ݶȡ�Ȩ�ؼ�ƫ����

tf.nn.sparse_softmax_cross_entropy_with_logits()�����label��ʽΪһά����;
tf.nn.softmax_cross_entropy_with_logits()�����label��ʽΪone_hot��ʽ��

���������������й��̿��Բ��Ϊ����:
1) softmax������ÿ������Ӧ�����������һ����ʹ��������ĺ�Ϊ1���������Ϊ���������
   ת��Ϊinput data����Ϊÿ�����ĸ��ʡ�
2) Cross_Entropy��Ϊloss function����loss��
'''
def get_loss(pred, label):
	"""
	  ������ʧ����������� Dice Loss �� Focal Loss��

	  ������
	  - pred: ���������Ԥ��������
	  - label: ��ʵ��ǩ������

	  ���أ�
	  - hybird_loss: ����� Dice Loss �� Focal Loss ����ʧֵ��
	  """
	# seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label) #��ϡ��
	# seg = tf.reduce_mean(seg_loss)
	# hybird_loss = seg
	dice = dice_loss(pred, label,axis=1)
	focal = focal_loss(pred, label, alpha = 0.8, gamma = 2)
	hybird_loss = dice+focal

	return hybird_loss
