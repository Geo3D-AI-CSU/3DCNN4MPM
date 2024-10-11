# encoding:gbk
from __future__ import division
import os
import re
import sys
import math
import pymssql
import argparse
import importlib
import numpy as np
import tensorflow as tf
from model import *

sess = tf.InteractiveSession() 												# 创建TensorFlow会话

parser = argparse.ArgumentParser() 											# 解析参数,用于命令行传递参数

parser.add_argument('--filelist', help='Path to training set ground truth (.csv)', required=True)
parser.add_argument('--num_input_channels', type=int, default=7)			# 输入特征通道数量
parser.add_argument('--num_classes', type=int, default=2) 					# 分类数量
parser.add_argument('--data_size', type=int, default=16) 					# 数据尺寸
'''
1个epoch = batch * batch_size
'''
parser.add_argument('--epochs', type=int, default=50) 						# 迭代次数50
parser.add_argument('--batch_size', type=int, default=71) 					# 每个batch的大小
'''
学习率更新（梯度下降策略，目的是加快收敛）：
decayed_learning_rate = learning_rate*decay_rate^(globle_step/decay_steps)
其中，learning_rate为根据经验设置的初始学习率；
     decay_rate为根据经验设置的衰减率系数；
     globle_step为当前训练轮次，epoch或者batch；
     decay_steps通定义衰减周期，跟参数staircase配合，可以在decay_step个训练轮次内
     保持学习率不变。
'''
parser.add_argument('--optimizer', default='adam')							 # 优化器(梯度下降)
parser.add_argument('--learning_rate', type=float, default=0.005)			 # 初始学习率
parser.add_argument('--decay_step', type=int, default=200000)				 # 衰减系数
parser.add_argument('--decay_rate', type=float, default=0.9) 				 # 衰减系数
'''
梯度下降前做归一化处理可以提高精度
参数设置与学习率相同。
'''
parser.add_argument('--bn_init_decay', type=float, default=0.5) 			 # 初始归一化参数
parser.add_argument('--bn_decay_rate', type=int, default=0.5) 				 # 衰减系数
parser.add_argument('--bn_decay_clip', type=float, default=0.99) 			 # 衰减系数
parser.add_argument('--results_path') 										 # 结果存储路径
parser.add_argument('--output_filelist', default='E://GCN//CNN//new_exp//pre1//output.txt', help='TXT filename, filelist, each line is an output for pixel')
parser.add_argument('--log_dir', default='log')								 # 运行信息存储路径

FLAGS = parser.parse_args() 												 # 实例化对象

#大写：全局变量
FILELIST = FLAGS.filelist
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
OUTPUT_FILELIST = FLAGS.output_filelist
LOG_DIR = FLAGS.log_dir

'''创建log文件夹'''
if not os.path.exists(LOG_DIR): 
	os.mkdir(LOG_DIR)
os.system('copy model.py %s' % (LOG_DIR)) 										# back up of model def
os.system('copy train.py %s' % (LOG_DIR))										# back up of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

'''获取数据列表,文件名存储在txt中'''
#获取数据列表
TRAIN_FILES = []
TEST_FILES = []
VALIDATION_FILES = []
for line in open(os.path.join(FILELIST, 'train.txt')):
	line = re.sub(r"\n","",line) 											     # 去掉字符串中的转义字符正则化好用
	TRAIN_FILES.append(line)
for line in open(os.path.join(FILELIST, 'test2.txt')):
	line = re.sub(r"\n","",line) 										         # 去掉字符串中的转义字符正则化好用
	TEST_FILES.append(line)
VALIDATION_FILES = TEST_FILES
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
	with tf.device('/gpu:0'):
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
		pred = get_model1(data, is_training, bn_decay=bn_decay)
		pred_softmax = tf.nn.softmax(pred)
		loss = get_loss(pred, label)
		tf.summary.scalar('loss', loss)
		
		#准确率
		correct = tf.equal(tf.argmax(pred, -1), tf.to_int64(label)) #逐元素比较
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
				optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
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
		   'pred_softmax': pred_softmax,
           'loss': loss,
	       'train_op': train_op,
           'merged': merged,
	       'step': batch}
	
	fout_out_filelist = open(OUTPUT_FILELIST, 'w')
	
	#进行epoch循环
	for epoch in range(EPOCHS):
		
		#在同一个位置刷新输出,用于可视化更加美观
		log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
		sys.stdout.flush()
		
		#不需要传占位符声明的参数，所以可以直接运行
		get_train(sess, ops, train_writer)
		#get_test(sess, ops, test_writer)
		
	get_validation(sess, ops)
		
def get_train(sess, ops, train_writer):
	
	is_training_train = True
	
	#将训练数据顺序打乱(防止过拟合)
	train_file_idxs = np.arange(0, len(TRAIN_FILES))
	np.random.shuffle(train_file_idxs)
	
	#输入的TRAIN_FILES是实现一个epoch的文件 
	num_batch = len(TRAIN_FILES) // BATCH_SIZE
	
	total_correct = 0.0 #总分类正确数
	total_seen = 0.0 #已遍历样本数
	loss_sum = 0.0 #总损失
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] #每个类别的个数
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] #分类正确的个数
	
	#根据索引获取数据列表,一维数组
	filelist = TRAIN_FILES
	
	for batch_idx in range(num_batch):
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]
		
		#读取数据
		data_train, label_train, _ = read_csv(FILELIST, filenames, BATCH_SIZE, DATA_SIZE, 
		                       DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS, is_training_train)
		
		feed_dict = {ops['data']: data_train,
		             ops['label']: label_train,
		             ops['is_training']: is_training_train}
		             
		summary, step, _, loss, pred = sess.run([ops['merged'], ops['step'], 
			ops['train_op'], ops['loss'], ops['pred']], feed_dict = feed_dict)
		
		train_writer.add_summary(summary, step)
		pred = np.argmax(pred, -1)
		correct = np.sum(pred == label_train)
		total_correct += correct
		total_seen += (BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		loss_sum += loss
		
		#计算平均类别准确率
		pred_cls = pred.reshape(-1)
		label = label_train.reshape(-1)
		for i in range(len(label)):
			mark = int(label[i])
			total_seen_class[mark] += 1
			if(mark == int(pred_cls[i])):
				total_correct_class[mark] += 1
		
	train_mean_loss = loss_sum / float(num_batch)
	train_accuracy = total_correct / float(total_seen)
	train_avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
	log_string(LOG_FOUT, 'train mean loss: %f' % (train_mean_loss))
	log_string(LOG_FOUT, 'train accuracy: %f' % (train_accuracy))
	log_string(LOG_FOUT, 'train total correct class1: %f' % (total_correct_class[0]))
	log_string(LOG_FOUT, 'train total correct class2: %f' % (total_correct_class[1]))
	log_string(LOG_FOUT, 'train total seen class1: %f' % (total_seen_class[0]))
	log_string(LOG_FOUT, 'train total seen class2: %f' % (total_seen_class[1]))
	log_string(LOG_FOUT, 'train avg class acc: %f' % (train_avg_class_acc))

def get_test(sess, ops, test_writer):
	
	is_training = False
	
	#输入的TEST_FILES是实现一个epoch的文件
	num_batch = len(TEST_FILES) // BATCH_SIZE
	
	total_correct = 0.0 #总分类正确数
	total_seen = 0.0 #已遍历样本数
	loss_sum = 0.0 #总损失
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] #每个类别的个数
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] #分类正确的个数
	
	#根据索引获取数据列表,一维数组
	filelist = TEST_FILES
	
	for batch_idx in range(num_batch):
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]
		
		#读取数据
		data, label, index = read_csv(FILELIST, filenames, BATCH_SIZE, DATA_SIZE, 
		                       DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS, is_training)
		
		feed_dict = {ops['data']: data,
		             ops['label']: label,
		             ops['is_training']: is_training}
		
		summary, step, loss, pred = sess.run([ops['merged'], ops['step'],
			ops['loss'], ops['pred']], feed_dict = feed_dict)
			
		test_writer.add_summary(summary, step)
		pred = np.argmax(pred, -1)
		correct = np.sum(pred == label)
		total_correct += correct
		total_seen += (BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		loss_sum += loss
		
		#计算平均类别准确率
		pred_cls = pred.reshape(-1)
		label = label.reshape(-1)
		for i in range(len(label)):
			mark = int(label[i])
			total_seen_class[mark] += 1
			if(mark == int(pred_cls[i])):
				total_correct_class[mark] += 1
		
		
	test_mean_loss = loss_sum / float(num_batch)
	test_accuracy = total_correct / float(total_seen)
	test_avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
	log_string(LOG_FOUT, 'test mean loss: %f' % (test_mean_loss))
	log_string(LOG_FOUT, 'test accuracy: %f' % (test_accuracy))
	log_string(LOG_FOUT, 'test total correct class1: %f' % (total_correct_class[0]))
	log_string(LOG_FOUT, 'test total correct class2: %f' % (total_correct_class[1]))
	log_string(LOG_FOUT, 'test total seen class1: %f' % (total_seen_class[0]))
	log_string(LOG_FOUT, 'test total seen class2: %f' % (total_seen_class[1]))
	log_string(LOG_FOUT, 'test avg class acc: %f' % (test_avg_class_acc))
	
	return test_mean_loss, test_accuracy, test_avg_class_acc
    
cursor = None
connect = None

def connectToSqlServer(serverName='OS-20200927EPJM', dataBase='sanhetun'):
	# 函数用于连接到 Microsoft SQL Server 数据库
    global cursor														# 声明全局变量 cursor，用于执行 SQL 查询
    global connect 														# 声明全局变量 connect，用于表示数据库连接
    connect = pymssql.connect(serverName, 'sa', '123456', dataBase) 	# 使用 pymssql.connect() 方法建立数据库连接
    if connect:
        print('connect success')
        cursor = connect.cursor()
    else:
        print('connect fail')
        
def get_validation(sess, ops):
	
	is_training = False

	# 输入的 VALIDATION_FILES 是实现一个 epoch 的文件
	num_batch = len(VALIDATION_FILES) // BATCH_SIZE
	print(str(num_batch))

	# 根据索引获取数据列表，一维数组
	filelist = VALIDATION_FILES
	
	for batch_idx in range(num_batch):
		print(batch_idx)
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]
		
		#读取数据
		data, label, index = read_csv('E://GCN//CNN//data//2_global_part//part//', filenames, BATCH_SIZE, DATA_SIZE, 
		                       DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS, is_training)
		                       
		feed_dict = {ops['data']: data,
		             ops['is_training']: is_training}
		# 运行神经网络模型获取预测结果
		pred, pred_softmax = sess.run([ops['pred'], ops['pred_softmax']], feed_dict = feed_dict)
		
		length = len(np.array(pred).reshape(-1))
		size = length / (2 * BATCH_SIZE)        # 因为pred中包含了ore和non-ore两个类别的预测结果，而BATCH_SIZE表示每个批次的样本数量
		
		pred_val = np.array(pred) 				# pred包含ore和non-ore两个的pred
		pred_softmax = np.array(pred_softmax)
		
		index = np.array(index)
		
		pred_val = pred_val.reshape(-1)
		int(size)
		'''表示每个样本的长度，2
		表示两个类别（ore和non - ore）。这是为了将一维数组重新还原为原始的形状。'''
		pred_val = pred_val.reshape(-1,int(size),2)
		pred_softmax = pred_softmax.reshape(-1)
		pred_softmax = pred_softmax.reshape(-1,int(size),2) #从外到内
		index = index.reshape(-1)
		'''，其中int(size)表示每个样本的长度，3表示三个维度的位置信息。'''
		index = index.reshape(-1,int(size),3)
		pred_val = pred_val.tolist()
		pred_softmax = pred_softmax.tolist()
		index = index.tolist()

		'''两个嵌套的循环遍历BATCH_SIZE个样本和每个样本的每个位置信息，将每个位置的信息和对应的预测结果插入数据库表中'''
		for i in range(BATCH_SIZE):
			for j in range(len(pred_val[i])):
				sql = []
				sql_input = ''
				sql.append(index[i][j][0])
				sql.append(index[i][j][1])
				sql.append(index[i][j][2])
				sql.append(pred_val[i][j][0])
				sql.append(pred_val[i][j][1])
				sql.append(pred_softmax[i][j][0])
				sql.append(pred_softmax[i][j][1])
				sql = tuple(sql)
				sql = str(sql)
				sql_input = "insert into post_test(I,J,K,no_ore,ore,pred_no_ore,pred_ore) values" + sql
				cursor.execute(sql_input)
				connect.commit()

connectToSqlServer()	
if __name__ == "__main__":
	train()
	LOD_FOUT.close()
	
