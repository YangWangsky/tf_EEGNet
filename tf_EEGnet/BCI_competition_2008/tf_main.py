import tensorflow as tf
import numpy as np
import time
import datetime
import os
from utils import load_data, reformatInput, iterate_minibatches, Leave_Subject_Out
from tf_EEGNet_my import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1234)
tf.set_random_seed(1)


timestamp = datetime.datetime.now().strftime('%Y-%m-%d.%H.%M')  # 2018-04-21.15.59
log_path = os.path.join("runs", timestamp)

file_path = '/mnt/disk1/HeHe/MI/'


# 定义网络的参数
batch_size = 32
dropout_rate = 0.5 # Dropout 的概率，输出的可能性
n_classes = 4
fs = 128    # Sample frequence
n_Channels = 22
n_Samples = 256

learning_rate = 1e-8    # Adam
decay_steps = 1*(7*288//batch_size)   # 2592/batch_size为一个完整epoch所含的batch数量，1个epoch衰减一次
optimizer = tf.train.AdamOptimizer

num_epochs = 50


def train(X_inputs, labels, fold, batch_size, num_epochs, subj_id, learning_rate_default, Optimizer):
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(X_inputs, labels, fold)
    # normalization between all data
    X_mean = X_train.mean()
    X_std = X_train.std()
    X_train = (X_train - X_mean)/X_std
    X_val = (X_val - X_mean)/X_std
    X_test = (X_test - X_mean)/X_std

    model = Model(batch_size, n_classes, dropout_rate, fs, n_Channels, n_Samples)
    clip_all_weights = tf.get_collection('max-norm')    # max_norm

    with tf.name_scope('Optimizer'):
        # learning_rate = learning_rate_default * Decay_rate^(global_steps/decay_steps)
        global_steps = tf.Variable(0, name="global_step", trainable=False)     # train了多少次，即经历了多少个batch
        learning_rate = tf.train.exponential_decay(     # 学习率衰减
            learning_rate_default,  # Base learning rate.
            global_steps,
            decay_steps,             # 实现的效果就是经历了多少个epoch更新一次学习率
            0.95,  # Decay rate.
            staircase=True)
        optimizer = Optimizer(learning_rate)    # GradientDescentOptimizer  AdamOptimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_op = optimizer.minimize(Model._loss, global_step=global_steps, var_list=Model.T_vars)
            grads_and_vars = optimizer.compute_gradients(model._loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_steps)

    
    # Output directory for models and summaries
    # os.path.abspath绝对路径
    out_dir = os.path.abspath(os.path.join(os.path.curdir, log_path, ('cnn_'+str(subj_id)) ))  # 选择不同的路径保存不同模型以及不同被试
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss, accuracy and learning_rate
    loss_summary = tf.summary.scalar('loss', model._loss)
    acc_summary = tf.summary.scalar('train_acc', model._acc)
    lr_summary = tf.summary.scalar('learning_rate', learning_rate)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, lr_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, tf.get_default_graph()) # sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, tf.get_default_graph())

    # Test summaries
    test_summary_op = tf.summary.merge([loss_summary, acc_summary])
    test_summary_dir = os.path.join(out_dir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, tf.get_default_graph())

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, 'cnn')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    print("Starting training...")
    total_start_time = time.time()
    best_validation_accu = 0

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        train_acc_epoch = []
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            # 训练集
            train_err = train_acc = train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):  # shuffle不选true是因为生成序号时已打乱
                inputs, targets = batch
                summary, _, pred, loss, acc = sess.run([train_summary_op, train_op, model.output, model._loss, model._acc], 
                    {model.X_inputs: inputs, model.y_inputs: targets, model.tf_is_training: True, model.tf_bn_training: True})
                sess.run(clip_all_weights)  # max_norm
                train_acc += acc
                train_err += loss   # 累加计算总损失
                train_batches += 1
                train_summary_writer.add_summary(summary, sess.run(global_steps))  # global_steps 不run是不能直接用的。。

            av_train_err = train_err / train_batches
            av_train_acc = train_acc / train_batches
            train_acc_epoch.append(av_train_acc)

            # 验证集
            summary, pred, av_val_err, av_val_acc = sess.run([dev_summary_op, model.output, model._loss, model._acc], 
                    {model.X_inputs: X_val, model.y_inputs: y_val, model.tf_is_training: False, model.tf_bn_training: False})
            dev_summary_writer.add_summary(summary, sess.run(global_steps))

             # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(av_train_err))
            print("  training accuracy:\t\t{:.2f} %".format(av_train_acc * 100))
            print("  validation loss:\t\t\t{:.6f}".format(av_val_err))
            print("  validation accuracy:\t\t\t{:.2f} %".format(av_val_acc * 100))
            
            # 测试集
            summary, pred, av_test_err, av_test_acc = sess.run([test_summary_op, model.output, model._loss, model._acc], 
                {model.X_inputs: X_test, model.y_inputs: y_test, model.tf_is_training: False, model.tf_bn_training: False})
            test_summary_writer.add_summary(summary, sess.run(global_steps))

            print("Final results:")
            print("  test loss:\t\t\t\t\t{:.6f}".format(av_test_err))
            print("  test accuracy:\t\t\t\t{:.2f} %".format(av_test_acc * 100))


            if av_val_acc > best_validation_accu:   # 当且只有验证集达到最大精度时，才保存测试集结果
                best_validation_accu = av_val_acc   # 即使用eraly_stoping的结果
                test_acc_val = av_test_acc
                # 保存效果较好的模型和参数  不同被试不同类型模式使用不同地址
                saver.save(sess, checkpoint_prefix, global_step=sess.run(global_steps))

        train_acc = train_batches = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):  # shuffle不选true是因为生成序号时已打乱
            inputs, targets = batch
            acc = sess.run(model._acc, 
                    {model.X_inputs: inputs, model.y_inputs: targets, model.tf_is_training: False, model.tf_bn_training: False})
            train_acc += acc
            train_batches += 1
        last_train_acc = train_acc / train_batches  # 最后的模型在训练集集上的mse 

        last_val_acc = av_val_acc
        last_test_acc = av_test_acc
        print('-'*50)
        print('Time in total:', time.time()-total_start_time)
        print("Best validation accuracy:\t\t{:.2f} %".format(best_validation_accu * 100))
        print("Test accuracy when got the best validation accuracy:\t\t{:.2f} %".format(test_acc_val * 100))
        print('-'*50)
        print("Last train accuracy:\t\t{:.2f} %".format(last_train_acc * 100))
        print("Last validation accuracy:\t\t{:.2f} %".format(last_val_acc * 100))
        print("Last test accuracy:\t\t\t\t{:.2f} %".format(last_test_acc * 100))

    train_summary_writer.close()
    dev_summary_writer.close()
    test_summary_writer.close()
    return [last_train_acc, best_validation_accu, test_acc_val, last_val_acc, last_test_acc]


def train_all_subject(num_epochs=1):
    # Leave-Subject-Out cross validation
    subj_nums, fold_pairs = Leave_Subject_Out()

    EEGs, labels = load_data(file_path=file_path)
    EEGs = EEGs.reshape(-1, n_Channels, n_Samples, 1)
    labels = labels.reshape(-1)
    print(EEGs.shape)
    print(labels.shape)

    print('*'*200)
    acc_buf = []
    for subj_id in range(subj_nums):
        print('-'*100)
        print('The subj_id', subj_id, '\t\t Training the ' + 'cnn' + ' Model...')
        acc_temp = train(EEGs, labels, fold_pairs[subj_id], batch_size=batch_size, num_epochs=num_epochs, subj_id=subj_id,
                            learning_rate_default=0.0001, Optimizer=optimizer)
        acc_buf.append(acc_temp)
        tf.reset_default_graph()
        print('Done!')
    acc_buf = (np.array(acc_buf)).T
    print('All subjects for ', 'cnn', ' are done!')
    print('train_acc:\t', acc_buf[0])
    print('val_acc:\t', acc_buf[1])
    print('test_acc:\t', acc_buf[2])
    print('Last_val_acc:\t', acc_buf[3])
    print('Last_test_acc:\t', acc_buf[4])
    np.savetxt('./Accuracy_' + 'cnn' + '.csv', acc_buf, fmt='%.4f', delimiter=',')




if __name__ == '__main__':
    train_all_subject(num_epochs)