# encoding:gbk
from __future__ import division
import re
import sys
import argparse
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from model import *


sess = tf.InteractiveSession() 										# ��������ʽ�Ự��session��

parser = argparse.ArgumentParser()									# ��������,���������д��ݲ���,����ʹ��������һ�����޸�ģ�ͳ�����

# '--'��ʾ�˲���������λ�ò���
# default ����˲�����Ĭ��ֵ������������е��õ�ʱ��û��ָ���˲�������ֵ����ô�����Ĭ�ϴ���ֵΪ�˲�������ֵ
# type ����˲�������������
# help�������ָ����һ���˲���˵�������ڿ����������е������ģ����������˻��߳�ʱ�������ǲ������������Ч
parser.add_argument('--filelist', help='Path to training set ground truth (.csv)', default= 'E:\\xzh_profile\\CNN_for_MPM_code\\data\\3train_data')
parser.add_argument('--num_input_channels', type=int, default=7) 	# ��������ͨ������
parser.add_argument('--num_classes', type=int, default=2) 			# ��������
parser.add_argument('--data_size', type=int, default=16) 			# ���ݳߴ�
parser.add_argument('--epochs', type=int, default=50) 				# ��������
parser.add_argument('--batch_size', type=int, default=71) 			# ÿ��batch�Ĵ�С
'''
ѧϰ�ʸ��£��ݶ��½����ԣ�Ŀ���Ǽӿ���������
decayed_learning_rate = learning_rate*decay_rate^(globle_step/decay_steps)
���У�learning_rateΪ���ݾ������õĳ�ʼѧϰ�ʣ�
     decay_rateΪ���ݾ������õ�˥����ϵ����
     globle_stepΪ��ǰѵ���ִΣ�epoch����batch��
     decay_stepsͨ����˥�����ڣ�������staircase��ϣ�������decay_step��ѵ���ִ���
     ����ѧϰ�ʲ��䡣
'''
parser.add_argument('--optimizer', default='adam') 					# �Ż���(�ݶ��½�)
parser.add_argument('--learning_rate', type=float, default=0.005) 	# ��ʼѧϰ��
parser.add_argument('--decay_step', type=int, default=200000)		# ˥��ϵ��
parser.add_argument('--decay_rate', type=float, default=0.9) 		# ˥��ϵ��
'''
�ݶ��½�ǰ����һ�����������߾���
����������ѧϰ����ͬ��
'''
parser.add_argument('--bn_init_decay', type=float, default=0.5) 	# ��ʼ��һ������
parser.add_argument('--bn_decay_rate', type=int, default=0.5) 		# ˥��ϵ��
parser.add_argument('--bn_decay_clip', type=float, default=0.99) 	# ˥��ϵ��
parser.add_argument('--results_path') 								# ����洢·��
parser.add_argument('--log_dir', default='log4') 					# ������Ϣ�洢·��

FLAGS = parser.parse_args() 										# ʵ��������args���������parse�ĳ��������������Ϊһ��python�е��ֵ�(dictionary)

#��д��ȫ�ֱ���
FILELIST = FLAGS.filelist 											# ������ֵ
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

'''����log�ļ���'''
if not os.path.exists(LOG_DIR):                                 # ���LOG_DIRĿ¼�Ƿ����
	   os.mkdir(LOG_DIR)                                        # ��������ڣ��򴴽���Ŀ¼
os.system('copy model.py %s' % (LOG_DIR)) 						# ����'model.py'�ļ�
os.system('copy train.py %s' % (LOG_DIR))						# ����'train.py'�ļ�
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')    # ������¼ѵ�����̵�log��־�ļ�
LOG_FOUT.write(str(FLAGS)+'\n')                                 # ��¼ѵ������ĳ�����

'''��ȡ�����б�,�ļ����洢��txt��'''
# ��ȡ�����б�
# �����������б����ڴ洢ѵ���Ͳ����ļ�������
TRAIN_FILES = []
TEST_FILES = []
for line in open(os.path.join(FILELIST, 'train4.txt')):			# ��ȡѵ���ļ��б� 'train4.txt',һ��5��txt�����۽�����֤ʵ���Խ���żȻ�����
	line = re.sub(r"\n","",line) 								# ȥ���ַ����еĻ��з����򻯺���
	TRAIN_FILES.append(line)									# ������������ӵ�ѵ���ļ��б�
for line in open(os.path.join(FILELIST, 'test4.txt')):			# ��ȡ�����ļ��б� 'test4.txt'
	line = re.sub(r"\n","",line) 								# ȥ���ַ����еĻ��з�
	TEST_FILES.append(line)										# ������������ӵ������ļ��б�
# ʮ��������֤���ֺ����ݼ�
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
		# ʹ�� TensorFlow �����Ĺ��������������Ĳ���ָ���� GPU �豸��ִ��
		# '/device:GPU:0' ��ʾ���䵽��һ�� GPU�����Ϊ0����
		# ռλ����Ϊ�����ȡ��������ǰ����ռ�(�÷�����TensorFlow��̬��ܵ��÷�)
		data, label = placeholder_inputs(BATCH_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)
		is_training = tf.placeholder(dtype=tf.bool)
		
		# ��ʼ������
		# ֪ͨ�Ż�����ÿ��ѵ��ʱ���ӡ�������������
		batch = tf.Variable(0) 																	# ����һ�� TensorFlow ���������ڵ�����������ʼΪ0����������£�
		learning_rate = get_learning_rate(LEARNING_RATE, batch, BATCH_SIZE,
										DECAY_STEP, DECAY_RATE)
		bn_decay = get_bn_decay(BN_INIT_DECAY, batch, BATCH_SIZE,
								BN_DECAY_STEP, BN_DECAY_RATE,
								BN_DECAY_CLIP)
		# ���� TensorFlow ��ժҪ��summary�����ڼ�¼������Ϣ������ TensorBoard ���ӻ�
		'''
		TensorBoard �� TensorFlow ��һ�����ӻ����ߣ�
		���ڿ��ӻ�ѵ�������еĸ�����Ϣ��
		��������ͼ����ʧ���ߡ�ѧϰ�����ߡ��ݶ�ֱ��ͼ�ȡ�
		���ṩ��һ��ֱ�۵Ľ��棬�������ѧϰ��ҵ�߸��õ���⡢���Ժ��Ż����ǵ�ģ�͡�
		'''
		tf.summary.scalar('learning_rate', learning_rate)
		tf.summary.scalar('bn_decay', bn_decay) 												# �ϲ�ͼ����Ϣ���Զ�����summary
		
		'''
		tf.summary.scalar()�÷���
		���ã�������summaryȫ�����浽���̣��Ա�tensorboard��ʾ��
		���н����Ժ�,��tjn�ļ�������������־�ļ���������
		cmd����л�����Ӧ���ļ����£�����tensorboard��
		"tensorboard --logdir='tjn�ļ�·��'"
		Ȼ����ҳ��������"localhost:6006",(��ַ�ڲ�ͬ�������Ͽ��ܻ᲻ͬ)
		'''
		
		# ������ģ�ͼ���ʧ����
		pred = get_model(data, is_training, bn_decay = bn_decay)								# ��ȡģ��Ԥ����
		loss = get_loss(pred, label)															# ����ģ����ʧ
		tf.summary.scalar('loss', loss) 														# ��¼��ʧֵ�������� TensorBoard �п��ӻ���ʧ����

		# ׼ȷ��
		correct = tf.equal(tf.argmax(pred, -1), tf.cast(label,tf.int64))
		correct = tf.cast(correct, tf.float32) 													# boolת��Ϊfloat32
		accuracy = tf.reduce_sum(correct) / float(BATCH_SIZE)
		tf.summary.scalar('accuracy', accuracy)
		
		'''
		�������в�������ѧϰ�ʡ�
		��������ʧ��ѧϰ����Ӧ����ʧ������ѧϰ�ʴ󣬽��������ĽǶ�Խ��
		��ʧ��С�������ķ���ҲС��ѧϰ�ʾ�С�����ǲ��ᳬ���Լ����趨��ѧϰ�ʡ�
		'''
		# ����ѡ����Ż������ͽ��г�ʼ���Ż�������
		if OPTIMIZER == 'momentum':
			optimizer = tf.train.RMSPropOptimizer(learning_rate,momentum=0.9, decay=0.9, epsilon=1e-10)
		elif OPTIMIZER == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate)
		# ʹ���Ż�����С����ʧ������������ȫ�ֲ�����global_step��һ����������ÿ���Ż�ʱ�Զ���1��
		train_op = optimizer.minimize(loss, global_step = batch)
		'''
		minimize���ڲ���������������(1)��������������ݶ� (2)���ݶȸ�����Щ������ֵ
		train_opΪ�Ż���
		'''
		
		# ʵ�������󣬱������ȡ���������
		saver = tf.train.Saver()
		'''
		����һ�����ڱ���ͻ�ԭ TensorFlow ģ�͵� Saver ����
		�����洢�� checkpoint �ļ������ļ�������һ��Ŀ¼�����е�ģ���ļ��б�
		��������Ӧ��ģ��ʱֱ����"ckpt = tf.train.get_checkpoint_state(model_save_path)"��á�
		'''
	
	# ����tf.Session�����㷽ʽ��GPU����CPU��
	config = tf.ConfigProto() 											# ʵ��������
	config.gpu_options.allow_growth = True								# ��̬�����Դ�
	config.allow_soft_placement = True 									# �Զ�ѡ�������豸
	config.gpu_options.per_process_gpu_memory_fraction = 1 				# GPU�ڴ�ռ��������
	config.log_device_placement = False 								# �����ն˴�ӡ��������������ĸ��豸������
	sess = tf.Session(config=config)
	
	# ������summaryȫ�����浽���̣��Ա�tensorboard��ʾ
	merged = tf.summary.merge_all()
	# ���ô洢·��
	train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
	test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
	
	# ��ʼ��δ��ʼ����ȫ�ֱ���
	init = tf.global_variables_initializer()
	# ��is_trainingռλ���д���ֵ
	sess.run(init, {is_training: True})
	
	# �ֵ䣬��Ϊ�ӿڴ���ѵ��������epochѭ����
	ops = {'data': data,																# �������ݵ�ռλ��������
		   'label': label, 																# ��ǩ���ݵ�ռλ��������
		   'is_training':is_training, 													# ��ʾ�Ƿ���ѵ��״̬��ռλ��������
		   'pred': pred,																# ģ�͵�Ԥ���������
           'loss': loss, 																# ��ʧ��������
	       'train_op': train_op,														# �Ż����Ĳ���������ִ�в�������
           'merged': merged,  															# ���� TensorBoard �Ļ��ܲ���
	       'step': batch}																# ��ʾ��ǰѵ��������������ռλ��
	
	# ����epochѭ��
	for epoch in range(EPOCHS):
		
		# ��ͬһ��λ��ˢ�����,���ڿ��ӻ���������
		log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
		sys.stdout.flush()																# ��������׼���������������������ն�
		
		# ����Ҫ��ռλ�������Ĳ��������Կ���ֱ������
		train_mean_loss, train_accuracy = get_train(sess, ops, train_writer)
		test_mean_loss, test_accuracy, test_avg_class_acc = get_test(sess, ops, test_writer)
		
		# ����ģ�ͣ�ÿ10������һ��
		if epoch % 10 == 0:
			save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
			log_string(LOG_FOUT, "Model saved in file: %s" % save_path)


		
def get_train(sess, ops, train_writer):
	'''
	    ѵ�����̺���

	    Parameters
	    ----------
	    sess: tf.Session
	        TensorFlow �Ự����
	    ops: dict
	        �������� TensorFlow �������ֵ䡣
	    train_writer: tf.summary.FileWriter
	        ����д��ѵ��ժҪ�� TensorFlow FileWriter��

	    Returns
	    -------
	    train_mean_loss: float
	        ѵ������ƽ����ʧ��
	    train_accuracy: float
	        ѵ������׼ȷ�ʡ�
	    '''
	is_training_train = True
	
	# ��ѵ������˳�����(��ֹ�����)
	train_file_idxs = np.arange(0, len(TRAIN_FILES))				# TRAIN_FILESΪ��txt�ж�ȡ���ļ��б�
	np.random.shuffle(train_file_idxs)
	
	# �����TRAIN_FILES��ʵ��һ��epoch���ļ�
	num_batch = len(TRAIN_FILES) // BATCH_SIZE						# �����ж��ٸ�batch
		
	total_correct = 0.0 											# �ܷ�����ȷ��
	total_seen = 0.0 												# �ѱ���������
	loss_sum = 0.0													# ����ʧ
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)]    		# ÿ�����ĸ��� ����һ������ NUM_CLASSES ��0.0Ԫ�ص��б�
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] 		# ������ȷ�ĸ��� ����һ������ NUM_CLASSES ��0.0Ԫ�ص��б�
	
	# ����������ȡ�����б�,һά����
	filelist = TRAIN_FILES

	for batch_idx in range(num_batch):								# ����ÿ��batch
		
		start_idx = batch_idx * BATCH_SIZE							# ���㵱ǰbatch���������ݼ��е���ʼ�ͽ�������
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]					# ��ȡ��ǰbatch��Ӧ���ļ��б�
		# ��ȡ����
		data_train, label_train = read_csv(FILELIST, filenames, BATCH_SIZE, DATA_SIZE,
		                       DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)

		# ����feed_dict�������ݺͱ�ǩ����TensorFlow����ͼ��
		feed_dict = {ops['data']: data_train,
		             ops['label']: label_train,
		             ops['is_training']: is_training_train}

		# ����TensorFlow����ͼ��ִ��ѵ����������ȡѵ�������е������Ϣ
		summary, step, _, loss, pred = sess.run([ops['merged'], ops['step'], 
			ops['train_op'], ops['loss'], ops['pred']], feed_dict = feed_dict)

		# ��ѵ�����̵�ժҪ��ӵ�TensorBoard��
		train_writer.add_summary(summary, step)
		# ���㵱ǰbatch�ķ���׼ȷ��
		pred = np.argmax(pred, -1)
		correct = np.sum(pred == label_train)
		total_correct += correct
		total_seen += (BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		loss_sum += loss
		# ����ÿ������׼ȷ��
		pred_cls = pred.reshape(-1)
		label = label_train.reshape(-1)
		for i in range(len(label)):
			mark = int(label[i])
			total_seen_class[mark] += 1
			if(mark == int(pred_cls[i])):
				total_correct_class[mark] += 1
	# ��������ѵ������ƽ����ʧ��׼ȷ��
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
	
	# �����TEST_FILES��ʵ��һ��epoch���ļ�
	num_batch = len(TEST_FILES) // BATCH_SIZE
	
	total_correct = 0.0 														# �ܷ�����ȷ��
	total_seen = 0.0 															# �ѱ���������
	loss_sum = 0.0																# ����ʧ
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] 						# ÿ�����ĸ���
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] 					# ������ȷ�ĸ���
	
	# ����������ȡ�����б�,һά����
	filelist = TEST_FILES
	
	for batch_idx in range(num_batch):
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]
		
		# ��ȡ����
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
		
		# ����ƽ�����׼ȷ��
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
	
if __name__ == "__main__": 													# ��������ڣ��ж��Ƿ��ڵ�ǰģ��ֱ������
	start_time = time.time()												# ��¼����ʼʱ��
	former_time =datetime.datetime.now()
	print("Former Time :",former_time)
	train()																	# ���� train ����ִ�г������߼�
	end_time = time.time()													# ��¼�������ʱ��
	execution_time = end_time - start_time									# �������ִ��ʱ��
	print("Execution Time: {:.2f} seconds".format(execution_time))
	current_time = datetime.datetime.now()
	print("Current Time :", current_time)									# �ر���־�ļ�
	LOG_FOUT.close()
	
