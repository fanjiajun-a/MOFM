import torch
import torch.nn as nn
import os


class MMDNN(nn.Module):
    def __init__(self, para):
        super(MMDNN, self).__init__()
        services = para['services_num']
        users = para['users_num']
        m_users = para['user_len']
        n_services = para['service_len']
        mf_k = para['dimension']

        self.drop_layer = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        # stage 1
        # =======================================================================================
        self.indil = nn.BatchNorm1d(n_services * 10)
        self.local = nn.BatchNorm1d(m_users * 10)
        self.glob = nn.BatchNorm1d(services + users)
        self.indil_local = nn.BatchNorm1d(m_users * 10 + n_services * 10)
        # =======================================================================================
        self.fc1_1 = nn.Linear(users + services + m_users * 10 + n_services * 10,
                               int((users + services + m_users * 10 + n_services * 10) / 2))
        self.bn1 = nn.BatchNorm1d(int((users + services + m_users * 10 + n_services * 10) / 2))
        self.fc1_2 = nn.Linear(int((users + services + m_users * 10 + n_services * 10) / 2),
                               int((users + services + m_users * 10 + n_services * 10) / 4))
        self.fc1_3 = nn.Linear(int((users + services + m_users * 10 + n_services * 10) / 4), (m_users + n_services) * 4)
        self.fc1_4 = nn.Linear((m_users + n_services) * 4, (m_users + n_services) * 4)
        # stage 2
        self.fc2_1 = nn.Linear(users + services + m_users * 10 + n_services * 10 + (m_users + n_services) * 4, int((
            users + services + m_users * 10 + n_services * 10 + (m_users + n_services) * 4) / 2))
        self.bn2 = nn.BatchNorm1d(
            int((users + services + m_users * 10 + n_services * 10 + (m_users + n_services) * 4) / 2))
        self.fc2_2 = nn.Linear(
            int((users + services + m_users * 10 + n_services * 10 + (m_users + n_services) * 4) / 2),
            int((users + services + m_users * 10 + n_services * 10 + (m_users + n_services) * 4) / 4))
        self.fc2_3 = nn.Linear(
            int((users + services + m_users * 10 + n_services * 10 + (m_users + n_services) * 4) / 4),
            mf_k * 20)
        self.fc2_4 = nn.Linear(mf_k * 20, mf_k * 20)
        # stage 3
        self.fc3_1 = nn.Linear(users + services + m_users * 10 + n_services * 10 + mf_k * 20,
                               int((users + services + m_users * 10 + n_services * 10 + mf_k * 20) / 3))
        self.bn3 = nn.BatchNorm1d(int((users + services + m_users * 10 + n_services * 10 + mf_k * 20) / 3))
        self.fc3_2 = nn.Linear(int((users + services + m_users * 10 + n_services * 10 + mf_k * 20) / 3),
                               int((users + services + m_users * 10 + n_services * 10 + mf_k * 20) / 5))
        self.fc3_3 = nn.Linear(int((users + services + m_users * 10 + n_services * 10 + mf_k * 20) / 5),
                               mf_k * 3)
        self.fc3_4 = nn.Linear(mf_k * 3, 1)

    def forward(self, x1, x2, x3):
        # 纵向
        indil_X3 = self.indil(x3)
        indil_X3 = self.drop_layer(indil_X3)
        local_X2 = self.local(x2)
        local_X2 = self.drop_layer(local_X2)
        I_L = torch.cat((indil_X3, local_X2), 1)
        globa_x1 = self.glob(x1)
        globa_x1 = self.drop_layer(globa_x1)
        globa_I_L = self.indil_local(I_L)
        I_L = self.drop_layer(globa_I_L)
        I_L_G = torch.cat((I_L, globa_x1), 1)
        # 横向
        ILG = torch.cat((x3, x2, x1), 1)
        I_LG = torch.cat((I_L, x1), 1)
        # stage 1
        x = self.relu(self.fc1_1(I_L_G))
        x = self.relu(self.fc1_2(x))
        x = self.relu(self.fc1_3(x))
        x = self.relu(self.fc1_4(x))
        # stage 2
        x = torch.cat((x, I_LG), dim=1)
        x = self.relu(self.fc2_1(x))
        x = self.relu(self.fc2_2(x))
        x = self.relu(self.fc2_3(x))
        x = self.relu(self.fc2_4(x))
        # stage 3
        x = torch.cat((x, ILG), dim=1)
        x = self.relu(self.fc3_1(x))
        x = self.relu(self.fc3_2(x))
        x = self.relu(self.fc3_3(x))
        x = self.relu(self.fc3_4(x))

        return x
      
      
def save_net(net, epoch=-1):
    if not os.path.isdir('./Result/NetParameter'):
        os.makedirs('./Result/NetParameter')
    if not os.path.isdir('./Result/TestResult'):
        os.makedirs('./Result/TestResult')
    if epoch == -1:
        torch.save(net.state_dict(), './Result/NetParameter/Net_best')
    else:
        file_path = './Result//NetParameter/Net_' + str(epoch)
        torch.save(net.state_dict(), file_path)
