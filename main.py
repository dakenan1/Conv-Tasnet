# %%
import os
import tensorflow as tf
import librosa
import numpy as np
import logging
import shutil
import json
import sys

from dataloader import TasNetDataLoader
from tasnet import TasNet
from utils import create_dir, print_num_of_trainable_parameters
from utils import read_log, mySetup

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
GPUs = [0,1]

TRAIN_SET_LEN = -1
VALID_SET_LEN = -1

FLAG_RETRAIN = True

def average_loss(tower_loss):
    return tf.reduce_mean(tower_loss)

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
 
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    args, logger = mySetup()
    
    if os.path.isdir(args.log_dir) and FLAG_RETRAIN==False:
        shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)
        json.dump(vars(args), open(args.arg_file, 'w'), indent=4)
        logger.addHandler(logging.FileHandler(args.log_file))
        logger.setLevel(logging.INFO)

    gpu_num = len(GPUs)
    cliped_batch_size = int(args.batch_size // gpu_num)

    with tf.device("/cpu:0"):
        global_step = tf.Variable(0, trainable=False, name="global_step")
        if args.mode == 'train':
            train_dataloader = TasNetDataLoader("train", args.data_dir,
                                                args.batch_size, args.sample_rate)
            valid_dataloader = TasNetDataLoader("valid", args.data_dir,
                                                    args.batch_size, args.sample_rate)
            # data_train = tf.placeholder(tf.float32, [args.batch_size, 3,
            #                                          4*8000])
            # data_valid = tf.placeholder(tf.float32, [args.batch_size, 3,
            #                                          4*8000])
            data_train = train_dataloader.get_next()
            data_valid = valid_dataloader.get_next()
        else:
            infer_dataloader = TasNetDataLoader("infer", args.data_dir,
                                                args.batch_size, args.sample_rate)
            # data_infer = tf.placeholder(tf.float32, [args.batch_size, 3,
            #                                          4*8000])
            data_infer = infer_dataloader.get_next()

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope, tf.device("/cpu:0"):
        layers = {
            "conv1d_encoder":
            # mod by jiaxp@20190625, replace activation relu with linear
            tf.keras.layers.Conv1D(
                filters=args.N,
                kernel_size=args.L,
                strides=args.L // 2,
                name="encode_conv1d"),
            "bottleneck":
            tf.keras.layers.Conv1D(args.B, 1, 1),
            "1d_deconv":
            tf.keras.layers.Dense(args.L, use_bias=False)
        }
        for i in range(2):
            layers["1x1_conv_decoder_{}".format(i)] = \
                tf.keras.layers.Conv1D(args.N, 1, 1)
        for r in range(args.R):
            for x in range(args.X):
                now_block = "block_{}_{}_".format(r, x)
                layers[now_block + "first_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=args.H, kernel_size=1)
                layers[now_block + "first_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1])
                layers[now_block + "second_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1])
                layers[now_block + "second_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=args.B, kernel_size=1)

        if args.mode == 'train':
            tower_grads = []
            tower_loss = []
            tower_valid_loss = []
            learning_rate = tf.placeholder(tf.float32, [])
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            for gpu_i in range(gpu_num):
                with tf.name_scope("tower_{}".format(GPUs[gpu_i])),\
                tf.device("/gpu:{}".format(GPUs[gpu_i])):
                    train_model = TasNet("train",
                                         data_train[gpu_i*cliped_batch_size:
                                                    (gpu_i+1)*cliped_batch_size],
                                         layers, 2, args.N,
                                         args.L, args.B, args.H, args.P, args.X,
                                         args.R, args.sample_rate)
                    cur_loss = train_model.loss
                    tf.get_variable_scope().reuse_variables()
                    gradients = opt.compute_gradients(cur_loss)
                    tower_grads.append(gradients)
                    tower_loss.append(cur_loss)

                    scope.reuse_variables()
                    valid_model = TasNet("valid",
                                         data_valid[gpu_i*cliped_batch_size:
                                                    (gpu_i+1)*cliped_batch_size],
                                         layers, 2, args.N,
                                         args.L, args.B, args.H, args.P, args.X,
                                         args.R, args.sample_rate)
                    tower_valid_loss.append(valid_model.loss)
                    

            gradients = average_gradients(tower_grads)
            model_loss = average_loss(tower_loss)
            valid_loss = average_loss(tower_valid_loss)
        else:
            with tf.name_scope("tower_0"),\
                    tf.device("/gpu:{}".format(GPUs[0])):
                infer_model = TasNet("infer", data_infer, layers, 2, args.N,
                                     args.L, args.B, args.H, args.P, args.X,
                                     args.R, args.sample_rate)

    print_num_of_trainable_parameters()
    trainable_variables = tf.trainable_variables()

    valid_sdr = read_log(args.log_file)

    if args.mode == 'train':
        # with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope, tf.device("/cpu:0"):
        #     learning_rate = tf.placeholder(tf.float32, [])
        #     opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # gradients = tf.gradients(train_model.loss, trainable_variables)
        # clip gradients
        # cliped_gradients = tf.clip_by_norm(gradients, 5)
        #     tower_grads = []
        #     tower_loss = []
        #     for gpu_i in range(gpu_num):
        #         with tf.name_scope("tower_{}".format(GPUs[gpu_i])),
        #         tf.device("/gpu:{}".format(GPUs[gpu_i])):
        #             cur_loss = train_model.loss
        #             tf.get_variable_scope().reuse_variables()
        #             gradients = opt.compute_gradients(cur_loss)
        #             tower_grads.append(gradients)
        #             tower_loss.append(cur_loss)
        # gradients = average_gradients(tower_grads)
                
        cliped_gradients = []
        v = []
        for i, (grad_x, v_x) in enumerate(gradients):
            cliped_gradients.append(tf.clip_by_norm(grad_x, 5))
            v.append(v_x)
        # cliped_gradients = tf.clip_by_norm(gradients, 5)
        
        # print variables
        # print("variables num: ", len(v))
        # for cur_v in v:
        #     print(cur_v)
        # print('*'*20)
        # trainable_variables = tf.trainable_variables()
        # print("variables num: ", len(trainable_variables))
        # for cur_v in trainable_variables:
        #     print(cur_v)
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            update = opt.apply_gradients(
                zip(cliped_gradients, v), global_step=global_step)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
    # with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(args.log_dir)
        if ckpt:
            logging.info('Loading model from %s', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logging.info('Loading model with fresh parameters')
            sess.run(tf.global_variables_initializer())

        if args.mode == 'train':
            lr = args.learning_rate
            valid_scores = [-1] * 2

            for epoch in range(1, args.max_epoch + 1):

                sess.run(train_dataloader.iterator.initializer)
                logging.info('-' * 20 + ' epoch {} '.format(epoch) + '-' * 25)

                train_iter_cnt, train_loss_sum = 0, 0
                while True:
                    try:
                        if TRAIN_SET_LEN > 0 and train_iter_cnt > TRAIN_SET_LEN:
                            raise(TypeError('123'))
                        # cur_sig = sess.run(train_dataloader.get_next())
                        cur_loss, _, cur_global_step =\
                            sess.run(
                                fetches=[model_loss, update, global_step],
                                feed_dict={learning_rate: lr})
                        train_loss_sum += cur_loss * args.batch_size
                        train_iter_cnt += args.batch_size
                        if train_iter_cnt % 100 == 0:
                            print(train_iter_cnt, '/', TRAIN_SET_LEN)
                    except (tf.errors.OutOfRangeError, TypeError) as a:
                        logging.info(
                            'step = {} , train SDR = {:5f} , lr = {:5f}'.
                            format(cur_global_step,
                                   -train_loss_sum / train_iter_cnt, lr))
                        break

                sess.run(valid_dataloader.iterator.initializer)
                valid_iter_cnt, valid_loss_sum = 0, 0
                while True:
                    try:
                        if VALID_SET_LEN > 0 and valid_iter_cnt > VALID_SET_LEN:
                            raise(TypeError('123'))
                        cur_loss, = sess.run([valid_loss])
                        valid_loss_sum += cur_loss * args.batch_size
                        valid_iter_cnt += args.batch_size
                    except (tf.errors.OutOfRangeError, TypeError) as a:
                        cur_sdr = -(valid_loss_sum / valid_iter_cnt)

                        valid_scores.append(cur_sdr)
                        if max(valid_scores[-3:]) < valid_sdr:
                            lr /= 2
                            # mod by jxp, run at least 3 epochs after lr is halved
                            valid_scores.append(valid_sdr)

                        logging.info('validation SDR = {:5f}'.format(cur_sdr))
                        if cur_sdr > valid_sdr:
                            valid_sdr = cur_sdr
                            saver.save(
                                sess,
                                args.checkpoint_path,
                                global_step=cur_global_step)
                        break
        else:
            sess.run(infer_dataloader.iterator.initializer)
            infer_iter_cnt, infer_loss_sum = 0, 0
            while True:
                try:
                    cur_loss, outputs, single_audios, cur_global_step = sess.run(
                        fetches=[
                            infer_model.loss, infer_model.outputs, infer_model.
                            single_audios, global_step
                        ])

                    now_dir = args.log_dir + "/test/" + str(
                        infer_iter_cnt) + "/"

                    create_dir(now_dir)

                    outputs = [np.squeeze(output) for output in outputs]
                    single_audios = [
                        np.squeeze(single_audio)
                        for single_audio in single_audios
                    ]

                    def write(inputs, filename):
                        librosa.output.write_wav(
                            now_dir + filename,
                            inputs,
                            args.sample_rate,
                            norm=True)

                    write(outputs[0], 's1.wav')
                    write(outputs[1], 's2.wav')
                    write(single_audios[0], 'true_s1.wav')
                    write(single_audios[1], 'true_s2.wav')

                    infer_loss_sum += cur_loss * args.batch_size
                    infer_iter_cnt += args.batch_size

                    if infer_iter_cnt % 100 == 0:
                        print(-infer_loss_sum / infer_iter_cnt, infer_iter_cnt)
                except tf.errors.OutOfRangeError:
                    logging.info('step = {} , infer SDR = {:5f}'.format(
                        cur_global_step, -infer_loss_sum / infer_iter_cnt))
                    break


#%%
