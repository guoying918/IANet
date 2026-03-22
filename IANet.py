import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
import time
import datetime
from tqdm import tqdm
import scipy.io as sio
from Tools.data_processing import *
from sklearn import metrics
from Tools import modelStatsRecord
import Tools.utils as utils
from Tools.metrics import MD_distance
import random
from models import *
torch.cuda.empty_cache()
SAVE_NAME = "results"

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-dataset","--dataset",type = str, default = 'IndianP') #PaviaU、Salinas、HanChuan、IndianP
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 16)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)  
parser.add_argument("-e","--episode",type = int, default= 5000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu", type = str, default = '1')
# target
parser.add_argument("-m","--test_class_num",type=int, default=16)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='1 2 3 4 5')

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMBEND_DIM = 128

args = parser.parse_args()
GPU = args.gpu
# Hyper Parameters
DATASET = args.dataset
CLASS_NUM = args.class_num
TEST_CLASS_NUM = args.test_class_num # the number of class

if DATASET == 'PaviaU':
    TAR_INPUT_DIMENSION = 103
    CLASS_NUM = TEST_CLASS_NUM = 9
elif DATASET == 'Salinas':
    TAR_INPUT_DIMENSION = 204
    CLASS_NUM = TEST_CLASS_NUM = 16
elif DATASET == 'IndianP':
    TAR_INPUT_DIMENSION = 200
    CLASS_NUM = TEST_CLASS_NUM = 16
elif DATASET == 'LongKou':
    TAR_INPUT_DIMENSION = 270
    CLASS_NUM = TEST_CLASS_NUM = 9
elif DATASET == 'HanChuan':
    TAR_INPUT_DIMENSION = 274
    CLASS_NUM = TEST_CLASS_NUM = 16

SRC_INPUT_DIMENSION = args.src_input_dim
N_DIMENSION = args.n_dim
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
# run 10 times
N_RUNS = 10
Query_nums = QUERY_NUM_PER_CLASS * TEST_CLASS_NUM

numComponents = 'without'
# FOLDER = './Datatsets/test_ocbs/'
FOLDER = '/home/guoying/Correlation-optimization/Datatsets/original_data/'

current_date = datetime.date.today().strftime("%Y%m%d")
# RESULT_DIR ='./' + current_date +'_result_multipatch/' + DATASET + '/'
RESULT_DIR ='./ResponseResults/' + DATASET + '/'

if not os.path.isdir(RESULT_DIR):
    os.makedirs(RESULT_DIR)
CHECKPOINT_PATH = "./checkpoints/"+ DATASET + "/"+ SAVE_NAME + "/"
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
CLASSIFICATIONMAP_PATH = "./classificationMaps/"+ DATASET + "/"+ SAVE_NAME + "/"
if not os.path.isdir(CLASSIFICATIONMAP_PATH):
    os.makedirs(CLASSIFICATIONMAP_PATH)

utils.same_seeds(0)

def data_read(path, file):
    # load source domain data set
    with open(os.path.join(path, file), 'rb') as handle:
        source_imdb = pickle.load(handle)
    source_imdb['data']=np.array(source_imdb['data'])
    source_imdb['Labels']=np.array(source_imdb['Labels'],dtype='int')
    source_imdb['set']=np.array(source_imdb['set'],dtype='int')

    # process source domain data set
    data_train = source_imdb['data'] # (86874, 9, 9, 100)
    labels_train = source_imdb['Labels'] # (86874,)
    keys_all_train = sorted(list(set(labels_train)))  # class [0,...,45]
    label_encoder_train = {}  #{0: 0, 1: 1, 2: 2, 3: 3,...,45: 45}
    for i in range(len(keys_all_train)):
        label_encoder_train[keys_all_train[i]] = i

    train_set = {}
    for class_, path in zip(labels_train, data_train):
        if label_encoder_train[class_] not in train_set:
            train_set[label_encoder_train[class_]] = []
        train_set[label_encoder_train[class_]].append(path)
    data = train_set
    del train_set
    del keys_all_train
    del label_encoder_train

    print("Num classes for source domain datasets: " + str(len(data)))
    print(data.keys())
    data = utils.sanity_check(data) # 200 labels samples per class
    print("Num classes of the number of class larger than 200: " + str(len(data))) # 40 classes  8000 samples

    for class_ in data:
        for i in range(len(data[class_])):
            image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
            data[class_][i] = image_transpose

    # source few-shot classification data
    metatrain_data = data
    print(len(metatrain_data.keys()), metatrain_data.keys()) # 40 classes
    del data
    return metatrain_data

sorted_files = 'Patch9_CKS_TRIAN_META_DATA.pickle'
path = '/data/local_userdata/guoying/trainall_ocbs'
base_datasets = data_read(path, sorted_files)

# loader targer datasets
if DATASET == 'Houston2018':
    Data_Band_Scaler, GroundTruth,_ = utils.data_load_TIFHDR_PCA(FOLDER, numComponents=10)
else:
    Data_Band_Scaler, GroundTruth = utils.load_data(DATASET, FOLDER)
print('Finished load dataset')

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

# crossEntropy = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
crossEntropy = nn.CrossEntropyLoss().cuda()

# run 10 times
nDataSet = N_RUNS
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TEST_CLASS_NUM])
P = np.zeros([nDataSet, TEST_CLASS_NUM])
k = np.zeros([nDataSet, 1])
training_time = np.zeros([nDataSet, 1])
test_time = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
latest_G,latest_RandPerm,latest_Row, latest_Column,latest_nTrain = None,None,None,None,None

seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]

for iDataSet in range(nDataSet):
    print('iDataSet--------',iDataSet)
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, G, RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS, query_num = QUERY_NUM_PER_CLASS)

    # network
    feature_encoder = feature_encode(SRC_INPUT_DIMENSION, TAR_INPUT_DIMENSION)
    print(get_parameter_number(feature_encoder))  # {'Total': 1519081, 'Trainable': 1519081}

    feature_encoder.apply(weights_init)
    
    feature_encoder.cuda()
    feature_encoder.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    # 余弦衰减
    # scheduler = CosineAnnealingLR(feature_encoder_optim, T_max=1000, eta_min=1e-4)

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    torch.cuda.synchronize()
    train_start = time.time()
    for episode in range(EPISODE):
        # source domain few-shot
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task_source = utils.Task(base_datasets, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)

            support_dataloader = utils.get_HBKC_data_loader(task_source, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task_source, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().__next__()
            querys, query_labels = query_dataloader.__iter__().__next__()

            support_features = feature_encoder(supports.cuda(), domain = "source")
            query_features = feature_encoder(querys.cuda(), domain = "source")

            logits = MD_distance(support_features, support_labels, query_features)
            loss = crossEntropy(logits, query_labels.cuda())

            # Update parameters
            feature_encoder_optim.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += query_labels.shape[0]
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task_target = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)

            support_dataloader = utils.get_HBKC_data_loader(task_target, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task_target, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            supports, support_labels = support_dataloader.__iter__().__next__()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().__next__()  # (75,100,9,9)

            support_features = feature_encoder(supports.cuda(), domain = "target")
            query_features = feature_encoder(querys.cuda(), domain = "target")
            
            logits = MD_distance(support_features, support_labels, query_features)
            loss = crossEntropy(logits, query_labels.cuda())

            # Update parameters
            feature_encoder_optim.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
        
            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += query_labels.shape[0]

        if (episode + 1) % 1000 == 0:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}: loss: {:6.4f}, query_sample_num: {:>3d}, acc {:6.4f}'.format(episode + 1, \
                                                                                                            loss.item(),
                                                                                                            query_labels.shape[0],
                                                                                                            total_hit / total_num))
        if (episode + 1) % 1000 == 0 or episode == 0:
            # test
            print("Testing ...")
            torch.cuda.synchronize()
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            test_labels_all = np.array([], dtype=np.int64)

            train_datas, train_labels = train_loader.__iter__().__next__()
            train_features = feature_encoder(train_datas.cuda(), domain='target')  # (45, 160)

            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0] # 128
                test_features = feature_encoder(test_datas.cuda(), domain = 'target')

                predict_logits = MD_distance(train_features, train_labels, test_features)
                predict_labels = torch.argmax(predict_logits, dim=1).cpu()
                
                total_rewards += np.sum([1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)])

                counter += batch_size
                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)
            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%) iDataSet: {} \n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset), iDataSet))    
            torch.cuda.synchronize()
            test_end = time.time()
            # pre_time = test_end - train_end
            # print('pre_time-------------', pre_time)

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),str(CHECKPOINT_PATH + "/feature_encoder_" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe =  episode

                acc[iDataSet] = total_rewards / len(test_loader.dataset)
                OA = acc[iDataSet]
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
                P[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    training_time[iDataSet] = train_end - train_start
    test_time[iDataSet] = test_end - train_end

    latest_G, latest_RandPerm, latest_Row, latest_Column, latest_nTrain = G, RandPerm, Row, Column, nTrain
    for i in range(len(predict)): 
        latest_G[latest_Row[latest_RandPerm[latest_nTrain + i]]][latest_Column[latest_RandPerm[latest_nTrain + i]]] = \
            predict[i] + 1
    sio.savemat(CLASSIFICATIONMAP_PATH + '/pred_map_latest' + '_' + str(iDataSet) + "iter_" + repr(int(OA * 10000)) + '.mat', {'latest_G': latest_G})

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')
###
ELEMENT_ACC_RES_SS4 = np.transpose(A)
AA_RES_SS4 = np.mean(ELEMENT_ACC_RES_SS4,0)
OA_RES_SS4 = np.transpose(acc)
KAPPA_RES_SS4 = np.transpose(k)
ELEMENT_PRE_RES_SS4 = np.transpose(P)
AP_RES_SS4= np.mean(ELEMENT_PRE_RES_SS4,0)
TRAINING_TIME_RES_SS4 = 0
TESTING_TIME_RES_SS4 = np.transpose(test_time)
classes_num = TEST_CLASS_NUM

outputs_chart = modelStatsRecord.OutputData(classes_num, N_RUNS)
for current_trial_turn in range(N_RUNS):
    outputs_chart.set_data('train_time', current_trial_turn, training_time[current_trial_turn])
    outputs_chart.set_data('predict_time', current_trial_turn, test_time[current_trial_turn])
    outputs_chart.set_data('AA', current_trial_turn, np.around(AA_RES_SS4[current_trial_turn] * 100, 2))
    outputs_chart.set_data('OA', current_trial_turn, np.around(acc[current_trial_turn] * 100, 2))
    outputs_chart.set_data('Kappa', current_trial_turn, np.around(k[current_trial_turn] * 100, 2))
    for i in range(1, classes_num + 1):
        outputs_chart.set_data(i, current_trial_turn, np.around(A[current_trial_turn][i - 1] * 100, 2))

if not os.path.isdir(RESULT_DIR):
    os.makedirs(RESULT_DIR)
SAVE_PATH = RESULT_DIR + str(N_RUNS) + SAVE_NAME + "_DIM" + str(EMBEND_DIM) + "_" + str(EPISODE) + "_" + str(TEST_LSAMPLE_NUM_PER_CLASS) + "shot" + '_' + str(SHOT_NUM_PER_CLASS) + ".xlsx"
xlsxname = 'New_work'
outputs_chart.output_data(SAVE_PATH, xlsxname)
