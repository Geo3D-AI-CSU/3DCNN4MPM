# encoding:gbk
from __future__ import division

import multiprocessing
import os
import re
import sys
import math
import argparse
import importlib
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import time
import datetime
from model_lightCNN import *
from multiprocessing import Process
#这个是使用我们的数据，用坤师姐模型跑的代码
# from tensorflow.python.client import timeline
# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()

sess = tf.InteractiveSession() #创建TensorFlow会话

parser = argparse.ArgumentParser() #解析参数,用于命令行传递参数

parser.add_argument('--filelist', help='Path to training set ground truth (.csv)', default= 'D:\\CNN_for_MPM_code\\3train_data')
# parser.add_argument('--filelist', help='Path to training set ground truth (.csv)', required=True, default= 'E:\\xzh_profile\\CNN_for_MPM_code\\data\\3train_data')
parser.add_argument('--num_input_channels', type=int, default=7) #输入特征通道数量
parser.add_argument('--num_classes', type=int, default=2) #分类数量
parser.add_argument('--data_size', type=int, default=16) #数据尺寸
'''
1个epoch = batch * batch_size
'''
'''需要修改的参数，徐章皓'''
parser.add_argument('--epochs', type=int, default=50) #迭代次数 mark
# parser.add_argument('--batch_size', type=int, default=71) #每个batch的大小 46 76 徐章皓修改
parser.add_argument('--batch_size', type=int, default=71) #每个batch的大小 46 76 徐章皓修改
'''  
学习率更新（梯度下降策略，目的是加快收敛）：
decayed_learning_rate = learning_rate*decay_rate^(globle_step/decay_steps)
其中，learning_rate为根据经验设置的初始学习率；
     decay_rate为根据经验设置的衰减率系数；
     globle_step为当前训练轮次，epoch或者batch；
     decay_steps通定义衰减周期，跟参数staircase配合，可以在decay_step个训练轮次内
     保持学习率不变。
'''
parser.add_argument('--optimizer', default='adam') #优化器(梯度下降)
parser.add_argument('--learning_rate', type=float, default=0.0001) #初始学习率0.005
parser.add_argument('--decay_step', type=int, default=200000) #衰减系数
parser.add_argument('--decay_rate', type=float, default=0.9) #衰减系数
'''
梯度下降前做归一化处理可以提高精度
参数设置与学习率相同。
'''
parser.add_argument('--bn_init_decay', type=float, default=0.5) #初始归一化参数
parser.add_argument('--bn_decay_rate', type=int, default=0.5) #衰减系数
parser.add_argument('--bn_decay_clip', type=float, default=0.99) #衰减系数
parser.add_argument('--results_path') #结果存储路径
parser.add_argument('--log_dir', default='Train_org1102') #运行信息存储路径

FLAGS = parser.parse_args() #实例化对象

#大写：全局变量
FILELIST = FLAGS.filelist #变量赋值
LEARNING_RATE = FLAGS.learning_rate
NUM_INPUT_CHANNELS = FLAGS.num_input_channels
NUM_CLASSES = FLAGS.num_classes
DATA_SIZE = FLAGS.data_size
EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = FLAGS.bn_init_decay
BN_DECAY_RATE = FLAGS.bn_decay_rate
BN_DECAY_CLIP = FLAGS.bn_decay_clip
BN_DECAY_STEP = DECAY_STEP
RESULTS_PATH = FLAGS.results_path
LOG_DIR = FLAGS.log_dir
# 这里也要修改 徐章皓
'''创建log文件夹'''
if not os.path.exists(LOG_DIR): 
	os.mkdir(LOG_DIR)
os.system('copy model.py %s' % (LOG_DIR)) #back up of model def
os.system('copy train.py %s' % (LOG_DIR)) #back up of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'train_org1102.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


'''获取数据列表,文件名存储在txt中'''
#获取数据列表
TRAIN_FILES = []
TEST_FILES = []
'''需要修改的参数，徐章皓'''
for line in open(os.path.join(FILELIST, 'train1.txt')):
	line = re.sub(r"\n","",line) #去掉字符串中的转义字符正则化好用
	TRAIN_FILES.append(line)
for line in open(os.path.join(FILELIST, 'test1.txt')):
	line = re.sub(r"\n","",line)
	TEST_FILES.append(line)
#十倍交叉验证划分后数据集
'''
数据集信息:用11880个csv文件
十倍交叉验证数据集信息:训练集10692
                   测试集1188
batch_size:训练集36
           测试集36
batch:训练集297
      测试集33
'''

def train():
	with tf.device('/device:GPU:0'):
		#with tf.Graph().as_default():
		#占位符，为后面读取的数据提前分配空间(该方法是TensorFlow静态框架的用法)
		data, label = placeholder_inputs(BATCH_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)
		is_training = tf.placeholder(dtype=tf.bool)

		#初始化参数
		#通知优化器在每次训练时增加“批处理”参数。
		batch = tf.Variable(0) #先做初始化，后续会更新
		learning_rate = get_learning_rate(LEARNING_RATE, batch, BATCH_SIZE,
										DECAY_STEP, DECAY_RATE)
		bn_decay = get_bn_decay(BN_INIT_DECAY, batch, BATCH_SIZE,
								BN_DECAY_STEP, BN_DECAY_RATE,
								BN_DECAY_CLIP)
		tf.summary.scalar('learning_rate', learning_rate)
		tf.summary.scalar('bn_decay', bn_decay) #合并图表信息，自动管理summary

		'''
		tf.summary.scalar()用法：
		作用：将所有summary全部保存到磁盘，以便tensorboard显示。
		运行结束以后,在tjn文件夹里面会产生日志文件保存结果。
		cmd命令，切换到相应的文件夹下，启动tensorboard。
		"tensorboard --logdir='tjn文件路径'"
		然后再页面上输入"localhost:6006",(地址在不同的主机上可能会不同)
		'''

		#定义卷积模型及损失函数
		pred = get_model(data, is_training, bn_decay = bn_decay)
		loss = get_loss(pred, label)
		sess=tf.Session()
		tf.summary.scalar('loss', loss)
		#准确率
		# correct = tf.equal(tf.argmax(pred, -1), tf.to_int64(label)) #逐元素比较  徐章皓修改
		correct = tf.equal(tf.argmax(pred, -1), tf.cast(label,tf.int64))
		correct = tf.cast(correct, tf.float32) #bool转换为float32
		accuracy = tf.reduce_sum(correct) / float(BATCH_SIZE)
		tf.summary.scalar('accuracy', accuracy)

		'''
		在运行中不断修正学习率。
		根据其损失量学习自适应，损失量大则学习率大，进行修正的角度越大，
		损失量小，修正的幅度也小，学习率就小，但是不会超过自己所设定的学习率。
		'''
		#优化方式选择
		if OPTIMIZER == 'momentum':
			#optimizer = tf.train.MomentumOptimizer(learning_rae, momentum=0.9)#MOMENTUM)
			optimizer = tf.train.RMSPropOptimizer(learning_rate,momentum=0.9, decay=0.9, epsilon=1e-10)
		elif OPTIMIZER == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate)
		#对参数进行修正
		train_op = optimizer.minimize(loss, global_step = batch)
		'''
		minimize的内部存在两个操作：(1)计算各个变量的梯度 (2)用梯度更新这些变量的值
		train_op为优化器
		'''

		#实例化对象，保存和提取神经网络参数
		saver = tf.train.Saver()
		'''
		参数存储在 checkpoint 文件，该文件保存了一个目录下所有的模型文件列表，
		后续进行应用模型时直接用"ckpt = tf.train.get_checkpoint_state(model_save_path)"获得。
		'''

	#配置tf.Session的运算方式（GPU或者CPU）
	config = tf.ConfigProto() #实例化对象
	config.gpu_options.allow_growth = True #动态申请显存
	config.allow_soft_placement = True #自动选择运行设备
	config.gpu_options.per_process_gpu_memory_fraction = 1 #GPU内存占用率设置
	config.log_device_placement = False #不在终端打印出各项操作是在哪个设备上运行
	sess = tf.Session(config=config)

	#将所有summary全部保存到磁盘，以便tensorboard显示
	merged = tf.summary.merge_all()
	#设置存储路径
	train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
	test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

	#初始化未初始化的全局变量
	init = tf.global_variables_initializer()
	#向is_training占位符中传入值
	sess.run(init, {is_training: True})

	#字典，作为接口传入训练和评估epoch循环中
	ops = {'data': data,
		   'label': label,
		   'is_training':is_training,
		   'pred': pred,
           'loss': loss,
	       'train_op': train_op,
           'merged': merged,
	       'step': batch}
	oreIdx = np.load('./patchesOre.npy')
	noOreIdx = np.load('./patchesNoOre.npy')
	seed = np.random.randint(1, 1000)
	np.random.seed(seed)
	np.random.shuffle(oreIdx)
	np.random.shuffle(noOreIdx)
	oreIdx = np.load('./patchesOre.npy')
	noOreIdx = np.load('./patchesNoOre.npy')
	seed = np.random.randint(1, 1000)
	np.random.seed(seed)
	np.random.shuffle(oreIdx)
	np.random.shuffle(noOreIdx)


	numDims = [296, 448, 32]
	trainIdxList, testIdxList = GetIdxsList(oreIdx[0:512, :], noOreIdx, 0.8, 4)
	# oreIdx：（ore_num,3）含矿元素坐标集；noOreidx不含矿元素坐标集；0.8训练集占比；n非含矿是含矿的n倍
	# allData=GetAllData('./trainForFc',numDims[0],numDims[1],numDims[2],NUM_INPUT_CHANNELS+1)
	allData = np.load('allData.npy')
	trainDataList, trainLabelList = GetFeedDictListFc(ops, allData, trainIdxList, True)
	testDataList, testlabelDataList = GetFeedDictListFc(ops, allData, testIdxList, False)

	#进行epoch循环
	for epoch in range(EPOCHS):
		#在同一个位置刷新输出,用于可视化更加美观
		log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
		sys.stdout.flush()
		#不需要传占位符声明的参数，所以可以直接运行
		t1=time.time()
		train_mean_loss, train_accuracy = get_train(trainDataList,trainLabelList,sess, ops, train_writer)
		test_mean_loss, test_accuracy, test_avg_class_acc = get_test(testDataList,testlabelDataList,sess, ops, test_writer)
		print("训练一个周期所需要的时间为：",time.time()-t1)
		#保存模型，每10个保存一次   修改过-徐章皓
		# if epoch % 10 == 0:
		if epoch % 10 == 0:
			save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
			log_string(LOG_FOUT, "Model saved in file: %s" % save_path)

		with open(os.path.join(LOG_DIR, 'train_org1102.csv'), 'a', newline='') as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow(
				[epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy])


def GetAllData(path, numI, numJ, numK, channel):
	allData = np.zeros((numI, numJ, numK, channel))
	filelist = os.listdir(path)
	num = 0
	for file in filelist:
		data = np.genfromtxt(("./trainForFc/" + file), delimiter=',', skip_header=1)
		num = num + 1
		for line in range(len(data)):
			i = int(data[line, 0] - 1)
			j = int(data[line, 1] - 1)
			k = int(data[line, 2] - 1)
			allData[i, j, k, 0:7] = data[line, 3:10]
			allData[i, j, k, 7] = data[line, -1]
		print("%d/%d" % (num, len(filelist)))
	np.save('allData.npy', allData)
	return allData


def GetIdxsList(oreIdx, noOreIdx, rate, n):
	# idxList = []
	# for num in numDims:
	# 	idxList.append(np.arange(8, num-8, dtype=np.int16))
	# idxsI, idxsJ, idxsK = np.meshgrid(idxList[0], idxList[1], idxList[2])
	# allIdxs = [idxsI.flatten(), idxsJ.flatten(), idxsK.flatten()]
	# allIdxs=np.array(allIdxs).T
	# seed = np.random.randint(1, 1000)
	# np.random.seed(seed)
	# np.random.shuffle(allIdxs)
	# numElements = allIdxs.shape[0]
	# mark = int(numElements*rate)
	len1 = oreIdx.shape[0]

	trainNum1 = int(rate * len1)
	testNum1 = len1 - trainNum1

	trainNum0 = n * trainNum1
	testNum0 = int((1 - rate) * trainNum0)

	train1 = oreIdx[0:trainNum1, :]
	train0 = noOreIdx[0:trainNum0, :]
	test1 = oreIdx[trainNum1:trainNum1 + testNum1, :]
	test0 = noOreIdx[trainNum0:trainNum0 + testNum0, :]

	train = np.vstack((train1, train0))
	test = np.vstack((test1, test0))

	seed = np.random.randint(1, 1000)
	np.random.seed(seed)
	np.random.shuffle(train)
	np.random.shuffle(test)
	return train, test


def GetFeedDictListFc(ops, allData, dataList, isTrain=True):
	totalNum = dataList.shape[0]
	num_batch = totalNum // BATCH_SIZE
	feedDictList = []
	labelDictList = []
	for batch_idx in range(num_batch):
		patch = np.zeros((BATCH_SIZE, 16, 16, 16, NUM_INPUT_CHANNELS))
		label = np.zeros((BATCH_SIZE, 16, 16, 16,))
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		batch_n = 0
		for patch_idx in range(start_idx, end_idx):
			if patch_idx >= totalNum:
				break
			i = dataList[patch_idx][0]
			j = dataList[patch_idx][1]
			k = dataList[patch_idx][2]
			patch[batch_n] = allData[i - 8:i + 8, j - 8:j + 8, k - 8:k + 8, 0:7]
			label[batch_n] = allData[i - 8:i + 8, j - 8:j + 8, k - 8:k + 8, -1]
			batch_n = batch_n + 1
		feedDict = {ops['data']: patch,
					ops['label']: label,
					ops['is_training']: isTrain}
		feedDictList.append(feedDict)
		labelDictList.append(label)
		print("batch %d/%d read_finished" % (batch_idx, num_batch))
	return feedDictList, labelDictList


def get_train(feedDictList,labelDictList,sess, ops, train_writer):

	#将训练数据顺序打乱(防止过拟合)
	numBatch=len(feedDictList)
	total_correct = 0.0 #总分类正确数
	total_seen = 0.0 #已遍历样本数
	loss_sum = 0.0 #总损失
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] #每个类别的个数
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] #分类正确的个数
	###修改读写
	for (feed_dict,label_train) in zip(feedDictList,labelDictList):
		summary, step, _, loss, pred = sess.run([ops['merged'], ops['step'],
			ops['train_op'], ops['loss'], ops['pred']], feed_dict = feed_dict)
		train_writer.add_summary(summary, step)
		pred = np.argmax(pred, -1)
		correct = np.sum(pred == label_train)
		total_correct += correct
		total_seen += (BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		loss_sum += loss
		# 打断点徐章皓
		#计算平均类别准确率
		pred_cls = pred.reshape(-1)
		label = label_train.reshape(-1)
		for i in range(len(label)):
			mark = int(label[i])
			total_seen_class[mark] += 1
			if(mark == int(pred_cls[i])):
				total_correct_class[mark] += 1
	train_mean_loss = loss_sum / float(numBatch)
	train_accuracy = total_correct / float(total_seen)
	train_avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
	log_string(LOG_FOUT, 'train mean loss: %f' % (train_mean_loss))
	log_string(LOG_FOUT, 'train accuracy: %f' % (train_accuracy))
	log_string(LOG_FOUT, 'train total correct class1: %f' % (total_correct_class[0]))
	log_string(LOG_FOUT, 'train total correct class2: %f' % (total_correct_class[1]))
	log_string(LOG_FOUT, 'train total seen class1: %f' % (total_seen_class[0]))
	log_string(LOG_FOUT, 'train total seen class2: %f' % (total_seen_class[1]))
	log_string(LOG_FOUT, 'train avg class acc: %f' % (train_avg_class_acc))

	return train_mean_loss, train_accuracy

def get_test(feedDictList,labelDictList,sess, ops, test_writer):
	#输入的TEST_FILES是实现一个epoch的文件
	numBatch =len(feedDictList)
	#print(len(TEST_FILES))
	#print(BATCH_SIZE)
	#print(num_batch)
	total_correct = 0.0 #总分类正确数
	total_seen = 0.0 #已遍历样本数
	loss_sum = 0.0 #总损失
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] #每个类别的个数
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] #分类正确的个数

	for feed_dict,label in zip(feedDictList,labelDictList):
		summary, step, loss, pred = sess.run([ops['merged'], ops['step'],
			     ops['loss'], ops['pred']], feed_dict = feed_dict)
		test_writer.add_summary(summary, step)
		pred = np.argmax(pred, -1)
		correct = np.sum(pred == label)
		total_correct += correct
		total_seen += (BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		loss_sum += (loss * BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		#计算平均类别准确率
		pred = pred.reshape(-1)
		label = label.reshape(-1)
		for i in range(len(label)):
			mark = int(label[i])
			total_seen_class[mark] += 1
			if(mark == int(pred[i])):
				total_correct_class[mark] += 1
	test_mean_loss = loss_sum / float(total_seen)
	test_accuracy = total_correct / float(total_seen)
	test_avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
	log_string(LOG_FOUT, 'test mean loss: %f' % (test_mean_loss))
	log_string(LOG_FOUT, 'test accuracy: %f'% (test_accuracy))
	log_string(LOG_FOUT, 'test total correct class1: %f' % (total_correct_class[0]))
	log_string(LOG_FOUT, 'test total correct class2: %f' % (total_correct_class[1]))
	log_string(LOG_FOUT, 'test total seen class1: %f' % (total_seen_class[0]))
	log_string(LOG_FOUT, 'test total seen class2: %f' % (total_seen_class[1]))
	log_string(LOG_FOUT, 'test avg class acc: %f' % (test_avg_class_acc))

	return test_mean_loss, test_accuracy, test_avg_class_acc
	
if __name__ == "__main__":
	train()
	LOG_FOUT.close()
	
