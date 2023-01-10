from math import sqrt
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
import sys
from model.MMDNN.MMDNN import MMDNN

def init_para():
    initial_para = {'net': 'MMDNN',
                    'dimension': 50,
                    'lambda': 30,
                    'maxIter': 300,
                    'batchSize': 512,
                    'epoch': 100,
                    'user_len': 25,
                    'service_len': 10,
                    'mutation rate': 0.4,
                    'dataset_path': 'DataPreprocessing/ws_dream/',
                    'qos_attribute': 'tp',
                    'random_state': 1,
                    'density': 0.025,
                    'confidence': 0.1,
                    'train_size': 0,
                    'train_set': 0.7,
                    'valid_set': 0.1,
                    'GPU_Parallel': False,
                    'device': 'cuda:0',
                    'learning rate': 0.01,
                    'threshold': 0.97
                    }
    return initial_para


def comput_result(prediction, target):
    test_vec_x = np.where(target > 0)
    prediction = prediction[test_vec_x]
    target = target[test_vec_x]
    error = []
    accuracy = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
        if 1-abs(target[i] - prediction[i])/target[i] >0 and 1-abs(target[i] - prediction[i])/target[i] <1:
             accuracy.append(1-abs(target[i] - prediction[i])/target[i])
    squared_error = []
    abs_error = []
    for val in error:
        squared_error.append(val * val)
        abs_error.append(abs(val))
    mse = float(sum(squared_error) / len(squared_error))
    mae = float(sum(abs_error) / len(abs_error))
    rmse = sqrt(sum(squared_error) / len(squared_error))
    nmae = mae / (max(target)-min(target))
    acc = float(sum(accuracy) /len(accuracy))
    return mse, mae, rmse, nmae, acc


def ini_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def dataset_result(dataset_loader, preprocessed_dataset):

    pred = preprocessed_dataset.predmat
    pred_result = []
    label_result = []
    for _input_data, _target in dataset_loader:
        pred_result.append(np.array([pred[item[0], item[1]] for item in _input_data]))
        label_result.append(np.array(_target))
    pred_result = np.concatenate(pred_result)
    label_result = np.concatenate(label_result)
    _, _MAE, _RMSE,_NMSE = comput_result(pred_result, label_result)
    return _MAE, _RMSE


def init_pred_net(para):

    if para['GPU_Parallel']:
        para['device'] = 'cuda:0'
    para['device'] = torch.device(para['device'] if torch.cuda.is_available() else "cpu")

    net = MMDNN(para).to(para['device'])

    criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    if para['GPU_Parallel']:
        para['device'] = 'cuda:0'
        if torch.cuda.device_count() > 1:
            print('Use', torch.cuda.device_count(), 'GPUs')
            net = nn.DataParallel(net)

    return net, criterion, optimizer


#
def net_test(input_dataloader, preprocessed_data, para, net):
    with torch.no_grad():
        output_ = []
        target_ = []
        for input_data, target in input_dataloader:
            user_mat = [preprocessed_data.user_load[item[0], :] for item in input_data]
            service_mat = [preprocessed_data.service_load[item[1], :] for item in input_data]
            user_tensor = torch.from_numpy(np.array(user_mat)).to(para['device'])
            service_tensor = torch.from_numpy(np.array(service_mat)).to(para['device'])
            out_put = net(user_tensor.unsqueeze(2), service_tensor.unsqueeze(2))
            output_.append(out_put.clone().detach_().cpu().numpy())
            target_.append(target.clone().detach_().cpu().numpy())
        output_ = np.concatenate(output_)
        target_ = np.concatenate(target_)
        _, _mae, _rmse, _nmse = comput_result(output_, target_)

    return _mae, _rmse, _nmse
