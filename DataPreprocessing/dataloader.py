import random
import numpy as np
from DataPreprocessing.NMF import NMF
import torch


class DATALOADER:
    def __init__(self, para, mode):
        if mode == 'train':
            print('[Start Train Data Preprocessing]\n')
        elif mode == 'test':
            print('[Start Test Data Preprocessing]\n')
        self.mode = mode
        self.para = para
        self.initial_qos = np.load(para['dataset_path'] + para['qos_attribute'] + 'Matrix.npy')
        self.distence = np.load(self.para['dataset_path'] + 'distence.npy')
        self.distence_services = np.load(self.para['dataset_path'] + 'distence_services.npy')
        self.distence_users = np.load(self.para['dataset_path'] + 'distence_user.npy')
        self.users_num, self.services_num = self.initial_qos.shape
        self.reliable_qos = self.initial_qos
        self.dataset = np.empty([1, 1])
        self.densi_qos = np.empty([1, 1])
        self.pcc_matrix1 = []
        self.pcc_matrix2 = []
        self.mf_U = np.empty([1, 1])
        self.mf_S = np.empty([1, 1])

    def unreliable_remove(self, confidence):
        posi = np.where(self.reliable_qos <= 0)
        unreliable_services = []
        for item in np.unique(posi[1]):
            count = (np.sum(posi[1] == item))
            if count > int(self.users_num * confidence):
                unreliable_services.append(item)
        self.reliable_qos = np.delete(self.reliable_qos, unreliable_services, axis=1)
        self.distence = np.delete(self.distence, unreliable_services, axis=1)
        self.distence_services = np.delete(self.distence_services, unreliable_services, axis=1)
        self.distence_services = np.delete(self.distence_services, unreliable_services, axis=0)
        posi = np.where(self.reliable_qos <= 0)
        unreliable_users = []
        for item in np.unique(posi[0]):
            count = (np.sum(posi[0] == item))
            if count > int(self.services_num * confidence):
                unreliable_users.append(item)
        self.reliable_qos = np.delete(self.reliable_qos, unreliable_users, axis=0)
        self.distence = np.delete(self.distence, unreliable_users, axis=0)
        self.distence_users = np.delete(self.distence_users, unreliable_users, axis=1)
        self.distence_users = np.delete(self.distence_users, unreliable_users, axis=0)
        self.users_num, self.services_num = self.reliable_qos.shape
        print('Unreliable users and services have been removed. (users:', unreliable_users.__len__(), 'services:',
              unreliable_services.__len__(), ')')

    def split(self, train_size=0):
        if train_size == 0:
            train_size = self.para['train_size']
        random.seed(self.para['random_state'])
        services_list = list(range(0, self.services_num))
        random.shuffle(services_list)
        users_list = list(range(0, self.users_num))
        random.shuffle(users_list)

        if self.mode == 'train':
            train_set = []
            services_train = services_list[0:int(self.services_num * train_size)]
            users_train = users_list[0:int(self.users_num * train_size)]
            for u in users_train:
                for s in services_train:
                    train_set.append(self.reliable_qos[u, s])
            train_set = np.array(train_set).reshape([len(users_train), len(services_train)])
            self.dataset = train_set
            self.users_num, self.services_num = train_set.shape
        elif self.mode == 'test':
            test_set = []
            services_test = services_list[int(self.services_num * train_size):]
            users_test = users_list[int(self.users_num * train_size):]
            for u in users_test:
                for s in services_test:
                    test_set.append(self.reliable_qos[u, s])
            test_set = np.array(test_set).reshape([len(users_test), len(services_test)])
            self.dataset = test_set
            self.users_num, self.services_num = test_set.shape
        else:
            print('..')

    def density(self, dens):
        random.seed(self.para['random_state'])
        list_len = self.services_num * self.users_num
        qos_list = list(range(0, list_len))
        random.shuffle(qos_list)
        qos_list_have = qos_list[0:int(list_len * dens)]
        qos_list_drop = qos_list[int(list_len * dens):]
        posi_have = [[int(i / self.services_num), int(i % self.services_num)] for i in qos_list_have]
        posi_drop = [[int(i / self.services_num), int(i % self.services_num)] for i in qos_list_drop]
        x_qos_lost = self.dataset.copy()
        for pos in posi_drop:
            x_qos_lost[pos[0], pos[1]] = 0
        posi_have = np.array(posi_have)
        posi_drop = np.array(posi_drop)
        self.densi_qos = x_qos_lost
        self.posi = (posi_have, posi_drop)
        self.pcc_matrix1, self.pcc_matrix2 = self.pcc(x_qos_lost,self.para)
        return x_qos_lost, (posi_have, posi_drop)

    def nmf(self, dimension):
        self.para['dimension'] = dimension
        U_train, S_train = NMF.predict(self.densi_qos, self.para)
        self.mf_U = U_train
        self.mf_S = S_train

    def pcc(self,data, para):
        services = self.services_num
        users = self.users_num
        users_num = int(para['user_graph_len'])
        services_num = int(para['service_graph_len'])

        pcc_matrix1 = np.corrcoef(data.T) * 0.5 + 0.5

        pcc_matrix2 = np.corrcoef(data) * 0.5 + 0.5
        pcc_matrix1 = np.argsort(pcc_matrix1)[(services - services_num):, :]
        pcc_matrix2 = np.argsort(pcc_matrix2)[(users - users_num):, :]
        return pcc_matrix1, pcc_matrix2

class Dataloader(DATALOADER):

    def __init__(self, para, mode):
        print('==========================================================================')
        super(Dataloader, self).__init__(para, mode)
        self.unreliable_remove(para['confidence'])
        self.split(para['train_size'])
        self.density(para['density'])
        self.nmf(para['dimension'])
        self.predmat = np.dot(self.mf_U, self.mf_S.T)
        para['services_num'] = self.services_num
        para['users_num'] = self.users_num
        print('==========================================================================')

    def pred_tensor(self, batch_data):

        batch_size = batch_data.shape[0]
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        for i in range(batch_size):
            data = batch_data[i]
            temp1 = self.compute_x1(data)
            temp2 = self.compute_x2(data)
            temp3 = self.compute_x3(data)
            temp4 = self.compute_x4(data)
            x1.append(temp1)
            x2.append(temp2)
            x3.append(temp3)
            x4.append(temp4)
        x1 = torch.cat(x1).view(batch_size, -1).float()
        x2 = torch.cat(x2).view(batch_size, -1)
        x3 = torch.cat(x3).view(batch_size, -1)
        x4 = torch.from_numpy(np.array(x4)).unsqueeze(1)

        return x1.to(torch.float32), x2.to(torch.float32), x3.to(torch.float32), x4.to(torch.float32)

    def compute_x1(self, data):
        x_qos = torch.from_numpy(self.predmat)
        x1 = torch.cat([x_qos[int(data[0]), :], x_qos[:, int(data[1])]], dim=0)
        return x1

    def compute_x2(self, data):
        x2 = []
        for u in self.pcc_matrix2[:,data[0]]:
            for s in self.pcc_matrix1[:,data[1]]:
                x2.append(self.predmat[u, s])
        x2 = torch.from_numpy(np.array(x2))
        return x2

    def compute_x3(self, data):
        p = self.mf_U[data[0], :]
        q = self.mf_S[data[1], :]
        x3 = np.concatenate([p, q])
        x3 = torch.from_numpy(np.array(x3))
        return x3

    def compute_x4(self, data):
        x4 = self.predmat[data[0], data[1]]
        return x4
