from DataPreprocessing.dataloader import Dataloader
from DataPreprocessing.dataset import tensor_dataloader
import torch
import numpy as np
from tqdm import trange
import sys
import func
from tqdm import tqdm
import time

def train(para):

    func.ini_seed(para['random_state'])
    preprocessed_dataload = Dataloader(para, 'test')
    train_dataloader, valid_dataloader, test_dataloader = tensor_dataloader(preprocessed_dataload, para)
    net, criterion, optimizer = func.init_pred_net(para)
    mae_min = 9999
    mse_min = 9999
    rmse_min = 9999
    loss_epoch = []
    for epoch in range(50):
        print('----------------------第{}轮----------------------------'.format(epoch+1))
        starTime1 = time.clock()
        pbar = tqdm(train_dataloader, file=sys.stdout)
        for data, target in pbar:
            input_tensor = preprocessed_dataload.pred_tensor(data)
            optimizer.zero_grad()
            out_put = net(input_tensor[0].to(para['device']),
                          input_tensor[1].to(para['device']),
                          input_tensor[2].to(para['device']))
            loss = criterion(out_put.to(torch.float32), target.view(-1, 1).to(para['device']).to(torch.float32))
            loss.backward()
            optimizer.step()
            if __name__ == '__main__':
                pbar.set_description("Training: epoch %d .loss=%f " % (epoch, loss))
        endTime1 = time.clock()
        print('=============训练time为=={}=========='.format(endTime1 - starTime1))
        starTime2 = time.clock()
        with torch.no_grad():
            output_ = []
            target_ = []
            if __name__ == '__main__':
                pbar = tqdm(test_dataloader, file=sys.stdout)
            else:
                pbar = test_dataloader
            for data, target in pbar:
                input_tensor = preprocessed_dataload.pred_tensor(data)
                out_put = net(input_tensor[0].to(para['device']),
                              input_tensor[1].to(para['device']),
                              input_tensor[2].to(para['device']))
                loss = criterion(out_put.to(torch.float32), target.view(-1, 1).to(para['device']).to(torch.float32))
                output_.append(out_put.cpu().numpy())
                target_.append(target.cpu().numpy())
                if __name__ == '__main__':
                    pbar.set_description("Testing: epoch %d . loss=%f" % (epoch, loss))

            output_ = np.concatenate(output_)
            target_ = np.concatenate(target_)
            MSE, MAE, RMSE, NMAE, ACC = func.comput_result(output_, target_)
            if __name__ == '__main__':
                print('Test Result:MSE=%f MAE=%f RMSE=%f NMAE=%f ACC=%f' % (MSE, MAE, RMSE, NMAE,ACC))
            loss_epoch.append([MSE, MAE, RMSE, NMAE,ACC])

            if MAE < mae_min:
                best_epoch = epoch
                mae_min = MAE
                mse_min = MSE
                rmse_min = RMSE
                nmae = NMAE
                acc =ACC
        endTime2 = time.clock()
        print('=============测试time为=={}=========='.format(endTime2-starTime2))
        print('=============总time为=={}=========='.format(endTime2-starTime1))
    if __name__ == '__main__':
        print('The best epoch is %d,The MAE is %f' % (best_epoch, mae_min))
    return [mae_min, mse_min, rmse_min, nmae, acc]


if __name__ == '__main__':
    initial_para = func.init_para()
    train(initial_para)
