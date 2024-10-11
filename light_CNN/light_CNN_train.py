# encoding:gbk
from __future__ import division
import re
import sys
import argparse
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from model import *


sess = tf.InteractiveSession() 										# 创建交互式会话（session）

parser = argparse.ArgumentParser()									# 解析参数,用于命令行传递参数,方便使用命令行一次性修改模型超参数

# '--'表示此参数是任意位置参数
# default 代表此参数的默认值，如果在命令行调用的时候没有指定此参数的数值，那么程序会默认此数值为此参数的数值
# type 代码此参数的数据类型
# help是你可以指定的一个此参数说明，后期可以用命令行调出查阅，对于其他人或者长时间编程忘记参数涵义的人有效
parser.add_argument('--filelist', help='Path to training set ground truth (.csv)', default= 'E:\\xzh_profile\\CNN_for_MPM_code\\data\\3train_data')
parser.add_argument('--num_input_channels', type=int, default=7) 	# 输入特征通道数量
parser.add_argument('--num_classes', type=int, default=2) 			# 分类数量
parser.add_argument('--data_size', type=int, default=16) 			# 数据尺寸
parser.add_argument('--epochs', type=int, default=50) 				# 迭代次数
parser.add_argument('--batch_size', type=int, default=71) 			# 每个batch的大小
'''
学习率更新（梯度下降策略，目的是加快收敛）：
decayed_learning_rate = learning_rate*decay_rate^(globle_step/decay_steps)
其中，learning_rate为根据经验设置的初始学习率；
     decay_rate为根据经验设置的衰减率系数；
     globle_step为当前训练轮次，epoch或者batch；
     decay_steps通定义衰减周期，跟参数staircase配合，可以在decay_step个训练轮次内
     保持学习率不变。
'''
parser.add_argument('--optimizer', default='adam') 					# 优化器(梯度下降)
parser.add_argument('--learning_rate', type=float, default=0.005) 	# 初始学习率
parser.add_argument('--decay_step', type=int, default=200000)		# 衰减系数
parser.add_argument('--decay_rate', type=float, default=0.9) 		# 衰减系数
'''
梯度下降前做归一化处理可以提高精度
参数设置与学习率相同。
'''
parser.add_argument('--bn_init_decay', type=float, default=0.5) 	# 初始归一化参数
parser.add_argument('--bn_decay_rate', type=int, default=0.5) 		# 衰减系数
parser.add_argument('--bn_decay_clip', type=float, default=0.99) 	# 衰减系数
parser.add_argument('--results_path') 								# 结果存储路径
parser.add_argument('--log_dir', default='log4') 					# 运行信息存储路径

FLAGS = parser.parse_args() 										# 实例化对象，args会包含所有parse的超参数，可以理解为一个python中的字典(dictionary)

#大写：全局变量
FILELIST = FLAGS.filelist 											# 变量赋值
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

'''创建log文件夹'''
if not os.path.exists(LOG_DIR):                                 # 检查LOG_DIR目录是否存在
	   os.mkdir(LOG_DIR)                                        # 如果不存在，则创建该目录
os.system('copy model.py %s' % (LOG_DIR)) 						# 备份'model.py'文件
os.system('copy train.py %s' % (LOG_DIR))						# 备份'train.py'文件
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')    # 创建记录训练过程的log日志文件
LOG_FOUT.write(str(FLAGS)+'\n')                                 # 记录训练输入的超参数

'''获取数据列表,文件名存储在txt中'''
# 获取数据列表
# 定义两个空列表用于存储训练和测试文件的内容
TRAIN_FILES = []
TEST_FILES = []
for line in open(os.path.join(FILELIST, 'train4.txt')):			# 读取训练文件列表 'train4.txt',一共5个txt，五折交叉验证实验以降低偶然性误差
	line = re.sub(r"\n","",line) 								# 去掉字符串中的换行符正则化好用
	TRAIN_FILES.append(line)									# 将处理后的行添加到训练文件列表
for line in open(os.path.join(FILELIST, 'test4.txt')):			# 读取测试文件列表 'test4.txt'
	line = re.sub(r"\n","",line) 								# 去除字符串中的换行符
	TEST_FILES.append(line)										# 将处理后的行添加到测试文件列表
# 十倍交叉验证划分后数据集
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
		# 使用 TensorFlow 上下文管理器，将后续的操作指定在 GPU 设备上执行
		# '/device:GPU:0' 表示分配到第一个 GPU（编号为0）上
		# 占位符，为后面读取的数据提前分配空间(该方法是TensorFlow静态框架的用法)
		data, label = placeholder_inputs(BATCH_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)
		is_training = tf.placeholder(dtype=tf.bool)
		
		# 初始化参数
		# 通知优化器在每次训练时增加“批处理”参数。
		batch = tf.Variable(0) 																	# 创建一个 TensorFlow 变量，用于迭代步数（初始为0，后续会更新）
		learning_rate = get_learning_rate(LEARNING_RATE, batch, BATCH_SIZE,
										DECAY_STEP, DECAY_RATE)
		bn_decay = get_bn_decay(BN_INIT_DECAY, batch, BATCH_SIZE,
								BN_DECAY_STEP, BN_DECAY_RATE,
								BN_DECAY_CLIP)
		# 创建 TensorFlow 的摘要（summary）用于记录标量信息，用于 TensorBoard 可视化
		'''
		TensorBoard 是 TensorFlow 的一个可视化工具，
		用于可视化训练过程中的各种信息，
		包括计算图、损失曲线、学习率曲线、梯度直方图等。
		它提供了一个直观的界面，帮助深度学习从业者更好地理解、调试和优化他们的模型。
		'''
		tf.summary.scalar('learning_rate', learning_rate)
		tf.summary.scalar('bn_decay', bn_decay) 												# 合并图表信息，自动管理summary
		
		'''
		tf.summary.scalar()用法：
		作用：将所有summary全部保存到磁盘，以便tensorboard显示。
		运行结束以后,在tjn文件夹里面会产生日志文件保存结果。
		cmd命令，切换到相应的文件夹下，启动tensorboard。
		"tensorboard --logdir='tjn文件路径'"
		然后再页面上输入"localhost:6006",(地址在不同的主机上可能会不同)
		'''
		
		# 定义卷积模型及损失函数
		pred = get_model(data, is_training, bn_decay = bn_decay)								# 获取模型预测结果
		loss = get_loss(pred, label)															# 计算模型损失
		tf.summary.scalar('loss', loss) 														# 记录损失值，用于在 TensorBoard 中可视化损失曲线

		# 准确率
		correct = tf.equal(tf.argmax(pred, -1), tf.cast(label,tf.int64))
		correct = tf.cast(correct, tf.float32) 													# bool转换为float32
		accuracy = tf.reduce_sum(correct) / float(BATCH_SIZE)
		tf.summary.scalar('accuracy', accuracy)
		
		'''
		在运行中不断修正学习率。
		根据其损失量学习自适应，损失量大则学习率大，进行修正的角度越大，
		损失量小，修正的幅度也小，学习率就小，但是不会超过自己所设定的学习率。
		'''
		# 根据选择的优化器类型进行初始化优化器对象
		if OPTIMIZER == 'momentum':
			optimizer = tf.train.RMSPropOptimizer(learning_rate,momentum=0.9, decay=0.9, epsilon=1e-10)
		elif OPTIMIZER == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate)
		# 使用优化器最小化损失函数，并更新全局步数（global_step是一个计数器，每次优化时自动加1）
		train_op = optimizer.minimize(loss, global_step = batch)
		'''
		minimize的内部存在两个操作：(1)计算各个变量的梯度 (2)用梯度更新这些变量的值
		train_op为优化器
		'''
		
		# 实例化对象，保存和提取神经网络参数
		saver = tf.train.Saver()
		'''
		创建一个用于保存和还原 TensorFlow 模型的 Saver 对象
		参数存储在 checkpoint 文件，该文件保存了一个目录下所有的模型文件列表，
		后续进行应用模型时直接用"ckpt = tf.train.get_checkpoint_state(model_save_path)"获得。
		'''
	
	# 配置tf.Session的运算方式（GPU或者CPU）
	config = tf.ConfigProto() 											# 实例化对象
	config.gpu_options.allow_growth = True								# 动态申请显存
	config.allow_soft_placement = True 									# 自动选择运行设备
	config.gpu_options.per_process_gpu_memory_fraction = 1 				# GPU内存占用率设置
	config.log_device_placement = False 								# 不在终端打印出各项操作是在哪个设备上运行
	sess = tf.Session(config=config)
	
	# 将所有summary全部保存到磁盘，以便tensorboard显示
	merged = tf.summary.merge_all()
	# 设置存储路径
	train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
	test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
	
	# 初始化未初始化的全局变量
	init = tf.global_variables_initializer()
	# 向is_training占位符中传入值
	sess.run(init, {is_training: True})
	
	# 字典，作为接口传入训练和评估epoch循环中
	ops = {'data': data,																# 输入数据的占位符或张量
		   'label': label, 																# 标签数据的占位符或张量
		   'is_training':is_training, 													# 表示是否在训练状态的占位符或张量
		   'pred': pred,																# 模型的预测输出张量
           'loss': loss, 																# 损失函数张量
	       'train_op': train_op,														# 优化器的操作，用于执行参数更新
           'merged': merged,  															# 用于 TensorBoard 的汇总操作
	       'step': batch}																# 表示当前训练步数的张量或占位符
	
	# 进行epoch循环
	for epoch in range(EPOCHS):
		
		# 在同一个位置刷新输出,用于可视化更加美观
		log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
		sys.stdout.flush()																# 立即将标准输出缓冲区的内容输出到终端
		
		# 不需要传占位符声明的参数，所以可以直接运行
		train_mean_loss, train_accuracy = get_train(sess, ops, train_writer)
		test_mean_loss, test_accuracy, test_avg_class_acc = get_test(sess, ops, test_writer)
		
		# 保存模型，每10个保存一次
		if epoch % 10 == 0:
			save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
			log_string(LOG_FOUT, "Model saved in file: %s" % save_path)


		
def get_train(sess, ops, train_writer):
	'''
	    训练过程函数

	    Parameters
	    ----------
	    sess: tf.Session
	        TensorFlow 会话对象。
	    ops: dict
	        包含所有 TensorFlow 操作的字典。
	    train_writer: tf.summary.FileWriter
	        用于写入训练摘要的 TensorFlow FileWriter。

	    Returns
	    -------
	    train_mean_loss: float
	        训练集的平均损失。
	    train_accuracy: float
	        训练集的准确率。
	    '''
	is_training_train = True
	
	# 将训练数据顺序打乱(防止过拟合)
	train_file_idxs = np.arange(0, len(TRAIN_FILES))				# TRAIN_FILES为从txt中读取的文件列表
	np.random.shuffle(train_file_idxs)
	
	# 输入的TRAIN_FILES是实现一个epoch的文件
	num_batch = len(TRAIN_FILES) // BATCH_SIZE						# 计算有多少个batch
		
	total_correct = 0.0 											# 总分类正确数
	total_seen = 0.0 												# 已遍历样本数
	loss_sum = 0.0													# 总损失
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)]    		# 每个类别的个数 创建一个包含 NUM_CLASSES 个0.0元素的列表
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] 		# 分类正确的个数 创建一个包含 NUM_CLASSES 个0.0元素的列表
	
	# 根据索引获取数据列表,一维数组
	filelist = TRAIN_FILES

	for batch_idx in range(num_batch):								# 遍历每个batch
		
		start_idx = batch_idx * BATCH_SIZE							# 计算当前batch在整个数据集中的起始和结束索引
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]					# 获取当前batch对应的文件列表
		# 读取数据
		data_train, label_train = read_csv(FILELIST, filenames, BATCH_SIZE, DATA_SIZE,
		                       DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)

		# 构建feed_dict，将数据和标签传入TensorFlow计算图中
		feed_dict = {ops['data']: data_train,
		             ops['label']: label_train,
		             ops['is_training']: is_training_train}

		# 运行TensorFlow计算图，执行训练操作，获取训练过程中的相关信息
		summary, step, _, loss, pred = sess.run([ops['merged'], ops['step'], 
			ops['train_op'], ops['loss'], ops['pred']], feed_dict = feed_dict)

		# 将训练过程的摘要添加到TensorBoard中
		train_writer.add_summary(summary, step)
		# 计算当前batch的分类准确率
		pred = np.argmax(pred, -1)
		correct = np.sum(pred == label_train)
		total_correct += correct
		total_seen += (BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		loss_sum += loss
		# 计算每个类别的准确率
		pred_cls = pred.reshape(-1)
		label = label_train.reshape(-1)
		for i in range(len(label)):
			mark = int(label[i])
			total_seen_class[mark] += 1
			if(mark == int(pred_cls[i])):
				total_correct_class[mark] += 1
	# 计算整个训练集的平均损失和准确率
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
	
	return train_mean_loss, train_accuracy
    
def get_test(sess, ops, test_writer):
	
	is_training = False
	
	# 输入的TEST_FILES是实现一个epoch的文件
	num_batch = len(TEST_FILES) // BATCH_SIZE
	
	total_correct = 0.0 														# 总分类正确数
	total_seen = 0.0 															# 已遍历样本数
	loss_sum = 0.0																# 总损失
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] 						# 每个类别的个数
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] 					# 分类正确的个数
	
	# 根据索引获取数据列表,一维数组
	filelist = TEST_FILES
	
	for batch_idx in range(num_batch):
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]
		
		# 读取数据
		data, label = read_csv(FILELIST, filenames, BATCH_SIZE, DATA_SIZE, 
		                       DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)
		
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
		loss_sum += (loss * BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		
		# 计算平均类别准确率
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
	
if __name__ == "__main__": 													# 主程序入口，判断是否在当前模块直接运行
	start_time = time.time()												# 记录程序开始时间
	former_time =datetime.datetime.now()
	print("Former Time :",former_time)
	train()																	# 调用 train 函数执行程序主逻辑
	end_time = time.time()													# 记录程序结束时间
	execution_time = end_time - start_time									# 计算程序执行时间
	print("Execution Time: {:.2f} seconds".format(execution_time))
	current_time = datetime.datetime.now()
	print("Current Time :", current_time)									# 关闭日志文件
	LOG_FOUT.close()
	
