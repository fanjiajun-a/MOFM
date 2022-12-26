import random
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class TensorDataset(Dataset):

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def tensor_dataloader(dataload, para):
    torch.manual_seed(para['random_state'])
    target = torch.from_numpy(dataload.dataset)
    target = target.reshape(-1)
    input_tenosr = [[u, s] for u in range(dataload.users_num) for s in range(dataload.services_num)]
    random.shuffle(input_tenosr)
    target_tensor = [target[item[0] * dataload.services_num + item[1]] for item in input_tenosr]
    length = int(len(input_tenosr) * para['train_set'])
    length1 = int(len(input_tenosr) * (para['train_set'] + para['valid_set']))

    train_input_tenosr = torch.from_numpy(np.array(input_tenosr[:length]))
    train_target_tensor = torch.from_numpy(np.array(target_tensor[:length]))
    train_dataset = TensorDataset(train_input_tenosr, train_target_tensor)
    train_loader = DataLoader(train_dataset,
                              batch_size=para['batchSize'],
                              shuffle=True,
                              num_workers=0)

    valid_input_tenosr = torch.from_numpy(np.array(input_tenosr[length:length1]))
    valid_target_tensor = torch.from_numpy(np.array(target_tensor[length:length1]))
    valid_dataset = TensorDataset(valid_input_tenosr, valid_target_tensor)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=para['batchSize'],
                              shuffle=True,
                              num_workers=0)

    test_input_tenosr = torch.from_numpy(np.array(input_tenosr[length1:]))
    test_target_tensor = torch.from_numpy(np.array(target_tensor[length1:]))
    test_dataset = TensorDataset(test_input_tenosr, test_target_tensor)
    test_loader = DataLoader(test_dataset,
                             batch_size=para['batchSize'],
                             shuffle=True,
                             num_workers=0)

    return train_loader, valid_loader, test_loader

