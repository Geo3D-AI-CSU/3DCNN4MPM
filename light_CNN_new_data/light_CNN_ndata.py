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
#�����ʹ�����ǵ����ݣ�����ʦ��ģ���ܵĴ���
# from tensorflow.python.client import timeline
# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()

sess = tf.InteractiveSession() #����TensorFlow�Ự

parser = argparse.ArgumentParser() #��������,���������д��ݲ���

parser.add_argument('--filelist', help='Path to training set ground truth (.csv)', default= 'D:\\CNN_for_MPM_code\\3train_data')
# parser.add_argument('--filelist', help='Path to training set ground truth (.csv)', required=True, default= 'E:\\xzh_profile\\CNN_for_MPM_code\\data\\3train_data')
parser.add_argument('--num_input_channels', type=int, default=7) #��������ͨ������
parser.add_argument('--num_classes', type=int, default=2) #��������
parser.add_argument('--data_size', type=int, default=16) #���ݳߴ�
'''
1��epoch = batch * batch_size
'''
'''��Ҫ�޸ĵĲ����������'''
parser.add_argument('--epochs', type=int, default=50) #�������� mark
# parser.add_argument('--batch_size', type=int, default=71) #ÿ��batch�Ĵ�С 46 76 ������޸�
parser.add_argument('--batch_size', type=int, default=71) #ÿ��batch�Ĵ�С 46 76 ������޸�
'''  
ѧϰ�ʸ��£��ݶ��½����ԣ�Ŀ���Ǽӿ���������
decayed_learning_rate = learning_rate*decay_rate^(globle_step/decay_steps)
���У�learning_rateΪ���ݾ������õĳ�ʼѧϰ�ʣ�
     decay_rateΪ���ݾ������õ�˥����ϵ����
     globle_stepΪ��ǰѵ���ִΣ�epoch����batch��
     decay_stepsͨ����˥�����ڣ�������staircase��ϣ�������decay_step��ѵ���ִ���
     ����ѧϰ�ʲ��䡣
'''
parser.add_argument('--optimizer', default='adam') #�Ż���(�ݶ��½�)
parser.add_argument('--learning_rate', type=float, default=0.0001) #��ʼѧϰ��0.005
parser.add_argument('--decay_step', type=int, default=200000) #˥��ϵ��
parser.add_argument('--decay_rate', type=float, default=0.9) #˥��ϵ��
'''
�ݶ��½�ǰ����һ������������߾���
����������ѧϰ����ͬ��
'''
parser.add_argument('--bn_init_decay', type=float, default=0.5) #��ʼ��һ������
parser.add_argument('--bn_decay_rate', type=int, default=0.5) #˥��ϵ��
parser.add_argument('--bn_decay_clip', type=float, default=0.99) #˥��ϵ��
parser.add_argument('--results_path') #����洢·��
parser.add_argument('--log_dir', default='Train_org1102') #������Ϣ�洢·��

FLAGS = parser.parse_args() #ʵ��������

#��д��ȫ�ֱ���
FILELIST = FLAGS.filelist #������ֵ
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
# ����ҲҪ�޸� �����
'''����log�ļ���'''
if not os.path.exists(LOG_DIR): 
	os.mkdir(LOG_DIR)
os.system('copy model.py %s' % (LOG_DIR)) #back up of model def
os.system('copy train.py %s' % (LOG_DIR)) #back up of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'train_org1102.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


'''��ȡ�����б�,�ļ����洢��txt��'''
#��ȡ�����б�
TRAIN_FILES = []
TEST_FILES = []
'''��Ҫ�޸ĵĲ����������'''
for line in open(os.path.join(FILELIST, 'train1.txt')):
	line = re.sub(r"\n","",line) #ȥ���ַ����е�ת���ַ����򻯺���
	TRAIN_FILES.append(line)
for line in open(os.path.join(FILELIST, 'test1.txt')):
	line = re.sub(r"\n","",line)
	TEST_FILES.append(line)
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
	with tf.device('/device:GPU:0'):
		#with tf.Graph().as_default():
		#ռλ����Ϊ�����ȡ��������ǰ����ռ�(�÷�����TensorFlow��̬��ܵ��÷�)
		data, label = placeholder_inputs(BATCH_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)
		is_training = tf.placeholder(dtype=tf.bool)

		#��ʼ������
		#֪ͨ�Ż�����ÿ��ѵ��ʱ���ӡ���������������
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

		#�������ģ�ͼ���ʧ����
		pred = get_model(data, is_training, bn_decay = bn_decay)
		loss = get_loss(pred, label)
		sess=tf.Session()
		tf.summary.scalar('loss', loss)
		#׼ȷ��
		# correct = tf.equal(tf.argmax(pred, -1), tf.to_int64(label)) #��Ԫ�رȽ�  ������޸�
		correct = tf.equal(tf.argmax(pred, -1), tf.cast(label,tf.int64))
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
			#optimizer = tf.train.MomentumOptimizer(learning_rae, momentum=0.9)#MOMENTUM)
			optimizer = tf.train.RMSPropOptimizer(learning_rate,momentum=0.9, decay=0.9, epsilon=1e-10)
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
		�����洢�� checkpoint �ļ������ļ�������һ��Ŀ¼�����е�ģ���ļ��б���
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
	# oreIdx����ore_num,3������Ԫ�����꼯��noOreidx������Ԫ�����꼯��0.8ѵ����ռ�ȣ�n�Ǻ����Ǻ����n��
	# allData=GetAllData('./trainForFc',numDims[0],numDims[1],numDims[2],NUM_INPUT_CHANNELS+1)
	allData = np.load('allData.npy')
	trainDataList, trainLabelList = GetFeedDictListFc(ops, allData, trainIdxList, True)
	testDataList, testlabelDataList = GetFeedDictListFc(ops, allData, testIdxList, False)

	#����epochѭ��
	for epoch in range(EPOCHS):
		#��ͬһ��λ��ˢ�����,���ڿ��ӻ���������
		log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
		sys.stdout.flush()
		#����Ҫ��ռλ�������Ĳ��������Կ���ֱ������
		t1=time.time()
		train_mean_loss, train_accuracy = get_train(trainDataList,trainLabelList,sess, ops, train_writer)
		test_mean_loss, test_accuracy, test_avg_class_acc = get_test(testDataList,testlabelDataList,sess, ops, test_writer)
		print("ѵ��һ����������Ҫ��ʱ��Ϊ��",time.time()-t1)
		#����ģ�ͣ�ÿ10������һ��   �޸Ĺ�-�����
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

	#��ѵ������˳�����(��ֹ�����)
	numBatch=len(feedDictList)
	total_correct = 0.0 #�ܷ�����ȷ��
	total_seen = 0.0 #�ѱ���������
	loss_sum = 0.0 #����ʧ
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] #ÿ�����ĸ���
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] #������ȷ�ĸ���
	###�޸Ķ�д
	for (feed_dict,label_train) in zip(feedDictList,labelDictList):
		summary, step, _, loss, pred = sess.run([ops['merged'], ops['step'],
			ops['train_op'], ops['loss'], ops['pred']], feed_dict = feed_dict)
		train_writer.add_summary(summary, step)
		pred = np.argmax(pred, -1)
		correct = np.sum(pred == label_train)
		total_correct += correct
		total_seen += (BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		loss_sum += loss
		# ��ϵ������
		#����ƽ�����׼ȷ��
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
	#�����TEST_FILES��ʵ��һ��epoch���ļ�
	numBatch =len(feedDictList)
	#print(len(TEST_FILES))
	#print(BATCH_SIZE)
	#print(num_batch)
	total_correct = 0.0 #�ܷ�����ȷ��
	total_seen = 0.0 #�ѱ���������
	loss_sum = 0.0 #����ʧ
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] #ÿ�����ĸ���
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] #������ȷ�ĸ���

	for feed_dict,label in zip(feedDictList,labelDictList):
		summary, step, loss, pred = sess.run([ops['merged'], ops['step'],
			     ops['loss'], ops['pred']], feed_dict = feed_dict)
		test_writer.add_summary(summary, step)
		pred = np.argmax(pred, -1)
		correct = np.sum(pred == label)
		total_correct += correct
		total_seen += (BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		loss_sum += (loss * BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		#����ƽ�����׼ȷ��
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
	