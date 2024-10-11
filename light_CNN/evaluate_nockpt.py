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

sess = tf.InteractiveSession() 												# ����TensorFlow�Ự

parser = argparse.ArgumentParser() 											# ��������,���������д��ݲ���

parser.add_argument('--filelist', help='Path to training set ground truth (.csv)', required=True)
parser.add_argument('--num_input_channels', type=int, default=7)			# ��������ͨ������
parser.add_argument('--num_classes', type=int, default=2) 					# ��������
parser.add_argument('--data_size', type=int, default=16) 					# ���ݳߴ�
'''
1��epoch = batch * batch_size
'''
parser.add_argument('--epochs', type=int, default=50) 						# ��������50
parser.add_argument('--batch_size', type=int, default=71) 					# ÿ��batch�Ĵ�С
'''
ѧϰ�ʸ��£��ݶ��½����ԣ�Ŀ���Ǽӿ���������
decayed_learning_rate = learning_rate*decay_rate^(globle_step/decay_steps)
���У�learning_rateΪ���ݾ������õĳ�ʼѧϰ�ʣ�
     decay_rateΪ���ݾ������õ�˥����ϵ����
     globle_stepΪ��ǰѵ���ִΣ�epoch����batch��
     decay_stepsͨ����˥�����ڣ�������staircase��ϣ�������decay_step��ѵ���ִ���
     ����ѧϰ�ʲ��䡣
'''
parser.add_argument('--optimizer', default='adam')							 # �Ż���(�ݶ��½�)
parser.add_argument('--learning_rate', type=float, default=0.005)			 # ��ʼѧϰ��
parser.add_argument('--decay_step', type=int, default=200000)				 # ˥��ϵ��
parser.add_argument('--decay_rate', type=float, default=0.9) 				 # ˥��ϵ��
'''
�ݶ��½�ǰ����һ�����������߾���
����������ѧϰ����ͬ��
'''
parser.add_argument('--bn_init_decay', type=float, default=0.5) 			 # ��ʼ��һ������
parser.add_argument('--bn_decay_rate', type=int, default=0.5) 				 # ˥��ϵ��
parser.add_argument('--bn_decay_clip', type=float, default=0.99) 			 # ˥��ϵ��
parser.add_argument('--results_path') 										 # ����洢·��
parser.add_argument('--output_filelist', default='E://GCN//CNN//new_exp//pre1//output.txt', help='TXT filename, filelist, each line is an output for pixel')
parser.add_argument('--log_dir', default='log')								 # ������Ϣ�洢·��

FLAGS = parser.parse_args() 												 # ʵ��������

#��д��ȫ�ֱ���
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

'''����log�ļ���'''
if not os.path.exists(LOG_DIR): 
	os.mkdir(LOG_DIR)
os.system('copy model.py %s' % (LOG_DIR)) 										# back up of model def
os.system('copy train.py %s' % (LOG_DIR))										# back up of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

'''��ȡ�����б�,�ļ����洢��txt��'''
#��ȡ�����б�
TRAIN_FILES = []
TEST_FILES = []
VALIDATION_FILES = []
for line in open(os.path.join(FILELIST, 'train.txt')):
	line = re.sub(r"\n","",line) 											     # ȥ���ַ����е�ת���ַ����򻯺���
	TRAIN_FILES.append(line)
for line in open(os.path.join(FILELIST, 'test2.txt')):
	line = re.sub(r"\n","",line) 										         # ȥ���ַ����е�ת���ַ����򻯺���
	TEST_FILES.append(line)
VALIDATION_FILES = TEST_FILES
#ʮ��������֤���ֺ����ݼ�
'''
���ݼ���Ϣ:��11880��csv�ļ�
ʮ��������֤���ݼ���Ϣ:ѵ����10692
                   ���Լ�1188
batch_size:ѵ����36
           ���Լ�36
batch:ѵ����297
      ���Լ�33
'''

def train():
	with tf.device('/gpu:0'):
		#with tf.Graph().as_default():
		#ռλ����Ϊ�����ȡ��������ǰ����ռ�(�÷�����TensorFlow��̬��ܵ��÷�)
		data, label = placeholder_inputs(BATCH_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)
		is_training = tf.placeholder(dtype=tf.bool)
		
		#��ʼ������
		#֪ͨ�Ż�����ÿ��ѵ��ʱ���ӡ�������������
		batch = tf.Variable(0) #������ʼ�������������
		learning_rate = get_learning_rate(LEARNING_RATE, batch, BATCH_SIZE,
										DECAY_STEP, DECAY_RATE)
		bn_decay = get_bn_decay(BN_INIT_DECAY, batch, BATCH_SIZE,
								BN_DECAY_STEP, BN_DECAY_RATE,
								BN_DECAY_CLIP)
		tf.summary.scalar('learning_rate', learning_rate)
		tf.summary.scalar('bn_decay', bn_decay) #�ϲ�ͼ����Ϣ���Զ�����summary
		
		'''
		tf.summary.scalar()�÷���
		���ã�������summaryȫ�����浽���̣��Ա�tensorboard��ʾ��
		���н����Ժ�,��tjn�ļ�������������־�ļ���������
		cmd����л�����Ӧ���ļ����£�����tensorboard��
		"tensorboard --logdir='tjn�ļ�·��'"
		Ȼ����ҳ��������"localhost:6006",(��ַ�ڲ�ͬ�������Ͽ��ܻ᲻ͬ)
		'''
		
		#������ģ�ͼ���ʧ����
		pred = get_model1(data, is_training, bn_decay=bn_decay)
		pred_softmax = tf.nn.softmax(pred)
		loss = get_loss(pred, label)
		tf.summary.scalar('loss', loss)
		
		#׼ȷ��
		correct = tf.equal(tf.argmax(pred, -1), tf.to_int64(label)) #��Ԫ�رȽ�
		correct = tf.cast(correct, tf.float32) #boolת��Ϊfloat32
		accuracy = tf.reduce_sum(correct) / float(BATCH_SIZE)
		tf.summary.scalar('accuracy', accuracy)
		
		'''
		�������в�������ѧϰ�ʡ�
		��������ʧ��ѧϰ����Ӧ����ʧ������ѧϰ�ʴ󣬽��������ĽǶ�Խ��
		��ʧ��С�������ķ���ҲС��ѧϰ�ʾ�С�����ǲ��ᳬ���Լ����趨��ѧϰ�ʡ�
		'''
		#�Ż���ʽѡ��
		if OPTIMIZER == 'momentum':
				optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
		elif OPTIMIZER == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate)
		#�Բ�����������
		train_op = optimizer.minimize(loss, global_step = batch)
		'''
		minimize���ڲ���������������(1)��������������ݶ� (2)���ݶȸ�����Щ������ֵ
		train_opΪ�Ż���
		'''
		
		#ʵ�������󣬱������ȡ���������
		saver = tf.train.Saver()
		'''
		�����洢�� checkpoint �ļ������ļ�������һ��Ŀ¼�����е�ģ���ļ��б�
		��������Ӧ��ģ��ʱֱ����"ckpt = tf.train.get_checkpoint_state(model_save_path)"��á�
		'''
	
	#����tf.Session�����㷽ʽ��GPU����CPU��
	config = tf.ConfigProto() #ʵ��������
	config.gpu_options.allow_growth = True #��̬�����Դ�
	config.allow_soft_placement = True #�Զ�ѡ�������豸
	config.gpu_options.per_process_gpu_memory_fraction = 1 #GPU�ڴ�ռ��������
	config.log_device_placement = False #�����ն˴�ӡ��������������ĸ��豸������
	sess = tf.Session(config=config)
	
	#������summaryȫ�����浽���̣��Ա�tensorboard��ʾ
	merged = tf.summary.merge_all()
	#���ô洢·��
	train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
	test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
	
	#��ʼ��δ��ʼ����ȫ�ֱ���
	init = tf.global_variables_initializer()
	#��is_trainingռλ���д���ֵ
	sess.run(init, {is_training: True})
	
	#�ֵ䣬��Ϊ�ӿڴ���ѵ��������epochѭ����
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
	
	#����epochѭ��
	for epoch in range(EPOCHS):
		
		#��ͬһ��λ��ˢ�����,���ڿ��ӻ���������
		log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
		sys.stdout.flush()
		
		#����Ҫ��ռλ�������Ĳ��������Կ���ֱ������
		get_train(sess, ops, train_writer)
		#get_test(sess, ops, test_writer)
		
	get_validation(sess, ops)
		
def get_train(sess, ops, train_writer):
	
	is_training_train = True
	
	#��ѵ������˳�����(��ֹ�����)
	train_file_idxs = np.arange(0, len(TRAIN_FILES))
	np.random.shuffle(train_file_idxs)
	
	#�����TRAIN_FILES��ʵ��һ��epoch���ļ� 
	num_batch = len(TRAIN_FILES) // BATCH_SIZE
	
	total_correct = 0.0 #�ܷ�����ȷ��
	total_seen = 0.0 #�ѱ���������
	loss_sum = 0.0 #����ʧ
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] #ÿ�����ĸ���
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] #������ȷ�ĸ���
	
	#����������ȡ�����б�,һά����
	filelist = TRAIN_FILES
	
	for batch_idx in range(num_batch):
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]
		
		#��ȡ����
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
		
		#����ƽ�����׼ȷ��
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
	
	#�����TEST_FILES��ʵ��һ��epoch���ļ�
	num_batch = len(TEST_FILES) // BATCH_SIZE
	
	total_correct = 0.0 #�ܷ�����ȷ��
	total_seen = 0.0 #�ѱ���������
	loss_sum = 0.0 #����ʧ
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] #ÿ�����ĸ���
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] #������ȷ�ĸ���
	
	#����������ȡ�����б�,һά����
	filelist = TEST_FILES
	
	for batch_idx in range(num_batch):
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]
		
		#��ȡ����
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
		
		#����ƽ�����׼ȷ��
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
	# �����������ӵ� Microsoft SQL Server ���ݿ�
    global cursor														# ����ȫ�ֱ��� cursor������ִ�� SQL ��ѯ
    global connect 														# ����ȫ�ֱ��� connect�����ڱ�ʾ���ݿ�����
    connect = pymssql.connect(serverName, 'sa', '123456', dataBase) 	# ʹ�� pymssql.connect() �����������ݿ�����
    if connect:
        print('connect success')
        cursor = connect.cursor()
    else:
        print('connect fail')
        
def get_validation(sess, ops):
	
	is_training = False

	# ����� VALIDATION_FILES ��ʵ��һ�� epoch ���ļ�
	num_batch = len(VALIDATION_FILES) // BATCH_SIZE
	print(str(num_batch))

	# ����������ȡ�����б�һά����
	filelist = VALIDATION_FILES
	
	for batch_idx in range(num_batch):
		print(batch_idx)
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]
		
		#��ȡ����
		data, label, index = read_csv('E://GCN//CNN//data//2_global_part//part//', filenames, BATCH_SIZE, DATA_SIZE, 
		                       DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS, is_training)
		                       
		feed_dict = {ops['data']: data,
		             ops['is_training']: is_training}
		# ����������ģ�ͻ�ȡԤ����
		pred, pred_softmax = sess.run([ops['pred'], ops['pred_softmax']], feed_dict = feed_dict)
		
		length = len(np.array(pred).reshape(-1))
		size = length / (2 * BATCH_SIZE)        # ��Ϊpred�а�����ore��non-ore��������Ԥ��������BATCH_SIZE��ʾÿ�����ε���������
		
		pred_val = np.array(pred) 				# pred����ore��non-ore������pred
		pred_softmax = np.array(pred_softmax)
		
		index = np.array(index)
		
		pred_val = pred_val.reshape(-1)
		int(size)
		'''��ʾÿ�������ĳ��ȣ�2
		��ʾ�������ore��non - ore��������Ϊ�˽�һά�������»�ԭΪԭʼ����״��'''
		pred_val = pred_val.reshape(-1,int(size),2)
		pred_softmax = pred_softmax.reshape(-1)
		pred_softmax = pred_softmax.reshape(-1,int(size),2) #���⵽��
		index = index.reshape(-1)
		'''������int(size)��ʾÿ�������ĳ��ȣ�3��ʾ����ά�ȵ�λ����Ϣ��'''
		index = index.reshape(-1,int(size),3)
		pred_val = pred_val.tolist()
		pred_softmax = pred_softmax.tolist()
		index = index.tolist()

		'''����Ƕ�׵�ѭ������BATCH_SIZE��������ÿ��������ÿ��λ����Ϣ����ÿ��λ�õ���Ϣ�Ͷ�Ӧ��Ԥ�����������ݿ����'''
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
	
