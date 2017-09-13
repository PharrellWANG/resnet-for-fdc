"""ResNet Train/Eval module.
"""
import time
import six
import sys

import data_input
import numpy as np
import resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10, cifar100 or fdc')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
													 'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '', 'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 8, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
													 'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 200,
														'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
												 'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
													 'Directory to keep the checkpoints. Should be a parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
														'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('block_size', 8,
														'block_size for fdc, can be 8, 16, 32 or 64')
tf.app.flags.DEFINE_integer('target_classes', 28, 'classes for fdc')
tf.app.flags.DEFINE_bool('DMM_included', False,
												 'is DMM included in the target classes')


def train(hps):
	"""Training loop."""
	with tf.device('/cpu:0'):
		images, labels = data_input.build_input(
			FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode,
			FLAGS.block_size)
	
	with tf.device('/gpu:0'):
		model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
		model.build_graph()
		
		param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
			tf.get_default_graph(),
			tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
		sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
		
		tf.contrib.tfprof.model_analyzer.print_model_analysis(
			tf.get_default_graph(),
			tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
		
		truth = tf.argmax(model.labels, axis=1)
		predictions = tf.argmax(model.predictions, axis=1)
		precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
		
		# export inference graph as pb format (proto)
		tf.train.write_graph(tf.get_default_graph(), FLAGS.train_dir,
												 'fdc_resnet_graph.pb', False)
		
		summary_hook = tf.train.SummarySaverHook(
			save_steps=50,
			# save_secs=120,
			output_dir=FLAGS.train_dir,
			summary_op=tf.summary.merge([model.summaries,
																	 tf.summary.scalar('Precision',
																										 precision)]))
		
		logging_hook = tf.train.LoggingTensorHook(
			tensors={'step'     : model.global_step,
							 'loss'     : model.cost,
							 'precision': precision},
			every_n_iter=100)
		
		class _LearningRateSetterHook(tf.train.SessionRunHook):
			"""Sets learning_rate based on global step."""
			
			def __init__(self):
				self._lrn_rate = 0.1
			
			def begin(self):
				self._lrn_rate = 0.1
			
			def before_run(self, run_context):
				return tf.train.SessionRunArgs(
					model.global_step,  # Asks for global step value.
					feed_dict={
						model.lrn_rate: self._lrn_rate})  # Sets learning rate
			
			def after_run(self, run_context, run_values):
				train_step = run_values.results
				if train_step < 40000:
					self._lrn_rate = 0.1
				elif train_step < 60000:
					self._lrn_rate = 0.01
				elif train_step < 80000:
					self._lrn_rate = 0.001
				else:
					self._lrn_rate = 0.0001
		
		with tf.train.MonitoredTrainingSession(
			checkpoint_dir=FLAGS.log_root,
			hooks=[logging_hook, _LearningRateSetterHook()],
			chief_only_hooks=[summary_hook],
			# Since we provide a SummarySaverHook, we need to disable default
			# SummarySaverHook. To do that we set save_summaries_steps to 0.
			save_summaries_steps=0,
			config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
			while not mon_sess.should_stop():
				mon_sess.run(model.train_op)


def evaluate(hps):
	"""Eval loop."""
	with tf.device('/cpu:0'):
		images, labels = data_input.build_input(
			FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode,
			FLAGS.block_size)
		model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
		model.build_graph()
		saver = tf.train.Saver()
		summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
		config = tf.ConfigProto(
			device_count={'GPU': 0}
		)
		sess = tf.Session(config=config)
		
		tf.train.start_queue_runners(sess)
		
		best_precision = 0.0
		while True:
			# time.sleep(2000)
			try:
				ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
			except tf.errors.OutOfRangeError as e:
				tf.logging.error('Cannot restore checkpoint: %s', e)
				continue
			if not (ckpt_state and ckpt_state.model_checkpoint_path):
				tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
				continue
			tf.logging.info('Loading checkpoint %s',
											ckpt_state.model_checkpoint_path)
			saver.restore(sess, ckpt_state.model_checkpoint_path)
			
			total_top_5, total_prediction, correct_prediction = 0, 0, 0
			correct_top_5_prediction = 0
			top_5_prediction_total = 0
			correct_top_8_prediction = 0
			top_8_prediction_total = 0
			correct_top_12_prediction = 0
			top_12_prediction_total = 0
			correct_top_16_prediction = 0
			top_16_prediction_total = 0
			correct_top_24_prediction = 0
			top_24_prediction_total = 0
			correct_top_28_prediction = 0
			top_28_prediction_total = 0
			
			if FLAGS.DMM_included:
				dmms_not_in_top_5 = 0
				dmms_not_in_top_8 = 0
				dmms_not_in_top_12 = 0
				dmms_not_in_top_16 = 0
				dmms_not_in_top_24 = 0
				dmms_not_in_top_28 = 0
			
			correct_in_range_one = 0
			correct_in_range_two = 0
			correct_in_range_three = 0
			correct_in_range_four = 0
			correct_in_range_five = 0
			
			loss = 0  # for eliminating IDE warning
			summaries = 0  # for eliminating IDE warning
			
			start = time.time()
			confusion_matrix_8x8 = np.zeros(
				(FLAGS.target_classes, FLAGS.target_classes))
			# noinspection PyUnresolvedReferences
			for _ in six.moves.range(FLAGS.eval_batch_count):
				(summaries, loss, predictions, truth, train_step) = sess.run(
					[model.summaries, model.cost, model.predictions,
					 model.labels, model.global_step])
				truth = np.argmax(truth, axis=1)
				
				for idx in range(hps.batch_size):
					row = truth[idx]
					col = predictions[idx]
					
					for_top_5 = col.argsort()[-5:][::-1]
					for_top_8 = col.argsort()[-8:][::-1]
					for_top_12 = col.argsort()[-12:][::-1]
					for_top_16 = col.argsort()[-16:][::-1]
					for_top_24 = col.argsort()[-24:][::-1]
					for_top_28 = col.argsort()[-28:][::-1]
					if row in for_top_5:
						correct_top_5_prediction += 1
					top_5_prediction_total += 1
					
					if row in for_top_8:
						correct_top_8_prediction += 1
					top_8_prediction_total += 1
					
					if row in for_top_12:
						correct_top_12_prediction += 1
					top_12_prediction_total += 1
					
					if row in for_top_16:
						correct_top_16_prediction += 1
					top_16_prediction_total += 1
					
					if row in for_top_24:
						correct_top_24_prediction += 1
					top_24_prediction_total += 1
					
					if row in for_top_28:
						correct_top_28_prediction += 1
					top_28_prediction_total += 1
					if FLAGS.DMM_included:
						if 35 not in for_top_5 and 36 not in for_top_5:
							dmms_not_in_top_5 += 1
						if 35 not in for_top_8 and 36 not in for_top_8:
							dmms_not_in_top_8 += 1
						if 35 not in for_top_12 and 36 not in for_top_12:
							dmms_not_in_top_12 += 1
						if 35 not in for_top_16 and 36 not in for_top_16:
							dmms_not_in_top_16 += 1
						if 35 not in for_top_24 and 36 not in for_top_24:
							dmms_not_in_top_24 += 1
						if 35 not in for_top_28 and 36 not in for_top_28:
							dmms_not_in_top_28 += 1
				
				predictions = np.argmax(predictions, axis=1)
				for idx in range(hps.batch_size):
					row = truth[idx]
					col = predictions[idx]
					confusion_matrix_8x8[row, col] += 1
					# print('index:  ' + str(idx) + '     correct_label: ' + str(row) + '     prediction: ' + str(col) +
					#       '     confusion_matrix_8x8[' + str(row) + ', ' + str(col) + '] = ' + str(confusion_matrix_8x8[row, col]))
					
					if col - 1 <= row <= col + 1:
						correct_in_range_one += 1
					
					if col - 2 <= row <= col + 2:
						correct_in_range_two += 1
					
					if col - 3 <= row <= col + 3:
						correct_in_range_three += 1
					
					if col - 4 <= row <= col + 4:
						correct_in_range_four += 1
					
					if col - 5 <= row <= col + 5:
						correct_in_range_five += 1
				
				correct_prediction += np.sum(truth == predictions)
				total_prediction += predictions.shape[0]
			
			# *********************************
			# confusion matrix #################
			# *********************************
			# for row in range(FLAGS.target_classes):
			#     print('---------------')
			#     print('mode : ' + str(row))
			#     print('----------')
			#     for col in range(FLAGS.target_classes):
			#         if confusion_matrix_8x8[row, col] != 0.0:
			#             print('mode: ' + str(row) + ' --->    number of predictions in mode ' + str(col) + ' :  ' + str(
			#                 confusion_matrix_8x8[row, col]))
			
			np.savetxt(
				"/Users/Pharrell_WANG/workspace/models/resnet/confusion_matrix"
				+ str(ckpt_state.model_checkpoint_path)[-10:] + ".csv",
				confusion_matrix_8x8, fmt='%i',
				delimiter=",")
			# confusion matrix #################
			precision = 1.0 * correct_prediction / total_prediction
			best_precision = max(precision, best_precision)
			
			precision_in_range_one = 1.0 * correct_in_range_one / total_prediction
			precision_in_range_two = 1.0 * correct_in_range_two / total_prediction
			precision_in_range_three = 1.0 * correct_in_range_three / total_prediction
			precision_in_range_four = 1.0 * correct_in_range_four / total_prediction
			precision_in_range_five = 1.0 * correct_in_range_five / total_prediction
			
			# avg_top_5 = total_top_5 / FLAGS.eval_batch_count
			top_5 = 1.0 * correct_top_5_prediction / top_5_prediction_total
			top_8 = 1.0 * correct_top_8_prediction / top_8_prediction_total
			top_12 = 1.0 * correct_top_12_prediction / top_12_prediction_total
			top_16 = 1.0 * correct_top_16_prediction / top_16_prediction_total
			top_24 = 1.0 * correct_top_24_prediction / top_24_prediction_total
			top_28 = 1.0 * correct_top_28_prediction / top_28_prediction_total
			if FLAGS.DMM_included:
				dmm_skipped_percent_for_top_5 = 1.0 * dmms_not_in_top_5 / top_5_prediction_total
				dmm_skipped_percent_for_top_8 = 1.0 * dmms_not_in_top_8 / top_8_prediction_total
				dmm_skipped_percent_for_top_12 = 1.0 * dmms_not_in_top_12 / top_12_prediction_total
				dmm_skipped_percent_for_top_16 = 1.0 * dmms_not_in_top_16 / top_16_prediction_total
				dmm_skipped_percent_for_top_24 = 1.0 * dmms_not_in_top_24 / top_24_prediction_total
				dmm_skipped_percent_for_top_28 = 1.0 * dmms_not_in_top_28 / top_28_prediction_total
			
			top_5_summ = tf.Summary()
			top_5_summ.value.add(
				tag='top_5', simple_value=top_5)
			summary_writer.add_summary(top_5_summ, train_step)
			
			top_8_summ = tf.Summary()
			top_8_summ.value.add(
				tag='top_8', simple_value=top_8)
			summary_writer.add_summary(top_8_summ, train_step)
			
			top_12_summ = tf.Summary()
			top_12_summ.value.add(
				tag='top_12', simple_value=top_12)
			summary_writer.add_summary(top_12_summ, train_step)
			
			top_16_summ = tf.Summary()
			top_16_summ.value.add(
				tag='top_16', simple_value=top_16)
			summary_writer.add_summary(top_16_summ, train_step)
			
			top_24_summ = tf.Summary()
			top_24_summ.value.add(
				tag='top_24', simple_value=top_24)
			summary_writer.add_summary(top_24_summ, train_step)
			
			top_28_summ = tf.Summary()
			top_28_summ.value.add(
				tag='top_28', simple_value=top_28)
			summary_writer.add_summary(top_28_summ, train_step)
			
			precision_summ = tf.Summary()
			precision_summ.value.add(
				tag='Precision', simple_value=precision)
			summary_writer.add_summary(precision_summ, train_step)
			
			best_precision_summ = tf.Summary()
			best_precision_summ.value.add(
				tag='Best Precision', simple_value=best_precision)
			summary_writer.add_summary(best_precision_summ, train_step)
			summary_writer.add_summary(summaries, train_step)
			tf.logging.info(
				'loss: %.3f, precision: %.3f, best precision: %.3f, top_5: %.3f, top_8: %.3f, top_12: %.3f, top_16: %.3f, top_24: %.3f, top_28: %.3f' %
				(loss, precision, best_precision, top_5, top_8, top_12, top_16,
				 top_24, top_28))
			# 'rg' means 'precision in range ...'
			# 'rg_1' in range +1 or -1
			tf.logging.info(
				'rg_1: %.3f, rg_2: %.3f, rg_3: %.3f, rg_4: %.3f, rg_5: %.3f' %
				(precision_in_range_one, precision_in_range_two,
				 precision_in_range_three, precision_in_range_four,
				 precision_in_range_five))
			if FLAGS.DMM_included:
				tf.logging.info(
					'DMM skipped percent-->>>>> for top_5: %.3f, top_8: %.3f, top_12: %.3f, top_16: %.3f, top_24: %.3f, top_28: %.3f' %
					(dmm_skipped_percent_for_top_5,
					 dmm_skipped_percent_for_top_8,
					 dmm_skipped_percent_for_top_12,
					 dmm_skipped_percent_for_top_16,
					 dmm_skipped_percent_for_top_24,
					 dmm_skipped_percent_for_top_28))
			# tf.logging.info(
			#     'loss: %.3f, precision: %.3f, best precision: %.3f' %
			#     (loss, precision, best_precision))
			summary_writer.flush()
			
			elapsed_time = time.time() - start
			print('total prediction: ' + str(total_prediction))
			print('single time spent for each prediction: ' + str(
				elapsed_time / float(total_prediction)))
			
			if FLAGS.eval_once:
				break
			
			time.sleep(180)


def main(_):
	# batch_size = 0
	# num_classes = 0
	
	if FLAGS.mode == 'train':
		batch_size = 128
	elif FLAGS.mode == 'eval':
		batch_size = 100
	else:
		raise ValueError('Only support two modes: train or eval')
	
	if FLAGS.dataset == 'cifar10':
		num_classes = 10
	elif FLAGS.dataset == 'cifar100':
		num_classes = 100
	elif FLAGS.dataset == 'fdc':
		num_classes = FLAGS.target_classes
	else:
		raise ValueError(
			'Only support three datasets: cifar10, cifar100 or fdc')
	
	hps = resnet_model.HParams(dataset_name=FLAGS.dataset,
														 batch_size=batch_size,
														 num_classes=num_classes,
														 min_lrn_rate=0.0001,
														 lrn_rate=0.1,
														 num_residual_units=5,
														 use_bottleneck=False,
														 weight_decay_rate=0.0002,
														 relu_leakiness=0.1,
														 optimizer='mom')
	
	# with tf.device(dev):
	if FLAGS.mode == 'train':
		train(hps)
	elif FLAGS.mode == 'eval':
		evaluate(hps)


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()
