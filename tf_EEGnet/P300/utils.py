import numpy as np
import os
import scipy.io
import scipy.signal


file_path = '/mnt/disk1/HeHe/MI/'
def preprocessing(file_path=file_path):
    fs = 2048 # 1048/0.5s    # sample frequency 2048Hz   有0.5s的数据
    num_subjs = 11   # there exits 11 subjects
    file_path_preproc = file_path + 'preproc'
    
    if not os.path.exists(file_path_preproc):
        os.makedirs(file_path_preproc)

    EEGs_Mat = []
    labels_Mat = []
    for id_subj in range(0,num_subjs):

        data_path = file_path + 'a0{}.mat'.format(id_subj+1)
        file = scipy.io.loadmat(data_path)
        EEGs = file['x'].transpose([2, 0, 1])
        labels = file['y'].reshape(-1)

        EEGs_Mat.append(EEGs)
        labels_Mat.append(labels.reshape(-1))
        print(id_subj)
    
    print('saving to {}/P300.mat'.format(file_path_preproc))
    scipy.io.savemat(file_path_preproc+'/P300.mat', {'EEGs':EEGs_Mat, 'labels':labels_Mat})



def load_data(file_path):
    file_path = file_path + '/preproc/P300_128Hz.mat'
    print("Loading data from %s" % (file_path))
    dataMat = scipy.io.loadmat(file_path)

    print("Data loading complete. Shape is: " , dataMat['EEGs'].shape)
    return dataMat['EEGs'], dataMat['labels']


def reformatInput(data, labels, indices):   # 划分训练集、验证集和测试集
    """
    Receives the indices for train and test datasets.
    param indices: tuple of (train, test) index numbers
    Outputs the train, validation, and test data and label datasets.
    """
    # indices[1]内的序号为某个subject的所有trial编号，indices[0]内为其它所有subject的编号
    trainIndices = indices[0][len(indices[1]):]     # [len(indices[1]):] 获取包括该序号之后的元素   训练集
    validIndices = indices[0][:len(indices[1])]     # [:len(indices[1])] 获取该序号之前的元素       验证集
    testIndices = indices[1]

    return [(data[trainIndices], labels[trainIndices]),
            (data[validIndices], labels[validIndices]),
            (data[testIndices], labels[testIndices])]


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """ # 用于生成batch
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images [n_samples, n_colors, W, H] and 5D numpy
                    array if working with sequence of images [n_timewindows, n_samples, n_colors, W, H].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    input_len = inputs.shape[0]
    assert input_len == len(targets)

    if shuffle:
        indices = np.arange(input_len)  
        np.random.shuffle(indices)      # 打乱索引号     # equal to indices = np.random.permutation(input_len)
    for start_idx in range(0, input_len, batchsize):
        # if start_idx + batchsize >= input_len:
        #     start_idx = input_len - batch_size        # 保证最后一个batch的size是batch_size
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            # inputs[excerpt]获取start_idx 到 start_idx + batchsize - 1，若最后序号达不到start_idx + batchsize - 1，则取到最后一个为止
        
        yield inputs[excerpt], targets[excerpt]



def Leave_Subject_Out(labels):

    subj_nums = 11
    trails_nums = []
    for i in range(subj_nums):
        trails_nums.append(len(labels[0][i].reshape(-1)))
     # trails_nums[i]为第i个被试具有的trial数量
    fold_pairs = []
    for i in range(subj_nums):
        tr = np.arange(np.sum(trails_nums))
        if i == 0:
            id_test = np.arange(trails_nums[0])
            tr[id_test]=0;      id_train = np.squeeze(np.nonzero(tr))
        elif i == (subj_nums-1):
            id_train = np.arange(np.sum(trails_nums[:-1]))
            tr[id_train]=0;     id_test = np.squeeze(np.nonzero(tr))
        else:
            id_test = np.arange(np.sum(trails_nums[:i]), np.sum(trails_nums[:i+1]))
            tr[id_test]=0;      id_train = np.squeeze(np.nonzero(tr))
        np.random.shuffle(id_test)  # Shuffle indices 打乱数据集的顺序
        np.random.shuffle(id_train)
        fold_pairs.append((id_train, id_test))
    return subj_nums, fold_pairs


# if __name__ == '__main__':
    