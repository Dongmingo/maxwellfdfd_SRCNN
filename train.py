import numpy as np
import argparse
import torch
import torch.nn as nn
from model_SRCNN import Net
import torch.optim as optim
from Data_augment import h5_DataAug
import os
import matplotlib.pyplot as plt

# set options : parser
parser = argparse.ArgumentParser(description='Pytorch SRCNN_md')
parser.add_argument('--numEpochs', type=int, default=500)
parser.add_argument('--LR', type=float, default=0.0005)
parser.add_argument('--scale', type=float, default=2)
parser.add_argument('--trainxfile', type=str, default='50samples/train_x.h5')
parser.add_argument('--trainyfile', type=str, default='50samples/train_y.h5')
parser.add_argument('--valxfile', type=str, default='20samples/val_x.h5')
parser.add_argument('--valyfile', type=str, default='20samples/val_y.h5')
parser.add_argument('--outputs-dir', type=str, default='saved_state')
parser.add_argument('--mode', default='client')
parser.add_argument('--port', default=61401)


# define main
def main():
    global opn, optimizer
    opn = parser.parse_args()
    print(opn)

    print('===> Loading Dataset')
    input, label, minmax = h5_DataAug(opn.trainxfile, opn.trainyfile, scale=opn.scale)
    Train_data = torch.transpose(torch.stack([input, label]), 0, 1)
    num_t_samples = list(Train_data.size())[0]

    input_v, label_v, minmax_v = h5_DataAug(opn.valxfile, opn.valyfile, scale=opn.scale)
    Val_data = torch.transpose(torch.stack([input_v, label_v]), 0, 1)
    num_v_samples = list(Val_data.size())[0]

    print('===> Building Models')
    model = Net()
    criterion = nn.MSELoss(reduction='mean')

    print('===> Setting Optimizer')
    optimizer = optim.Adam(model.parameters(), lr=opn.LR)

    epoch = 0
    print('===> Start Training')

    def PSNR(loss, minmax):
        psnr = 10 * np.log10((minmax ** 2) / loss)
        return psnr

    def train(Train_data, optimizer, model, criterion, epoch, train_loss, eval_loss):
        model.train()

        if epoch == 0:
            st_loss = 0
            psnr_total = 0

            for iteration, batch in enumerate(Train_data):
                input = torch.stack([batch[0]])
                label = torch.stack([batch[1]])
                [mi, Ma] = minmax[iteration]
                denorm_input = input * (Ma - mi) + mi
                denorm_label = label * (Ma - mi) + mi

                loss = criterion(denorm_input, denorm_label)
                st_loss += loss.item()
                psnr_total += PSNR(loss.item(), Ma - mi)

            print('==> Starting : zoomed by kron loss: {:.10f}, PSNR: {:.10f}'.format((st_loss / num_t_samples)**0.5,
                                                                                      psnr_total / num_t_samples))


        else:

            print('-----Epoch:[{}/{}], Learning_Rate : {}-----'.format(epoch, opn.numEpochs, opn.LR))
            total_loss = 0
            psnr_total = 0

            for iteration, batch in enumerate(Train_data):
                optimizer.zero_grad()
                input = torch.stack([batch[0]])
                label = torch.stack([batch[1]])
                output = model(input)
                # output denormalize를 위한 term
                xgrad_output = torch.tensor(output)
                [mi, Ma] = minmax[iteration]

                denorm_output = xgrad_output * (Ma - mi) + mi
                denorm_label = label * (Ma - mi) + mi

                loss = criterion(output, label)
                denorm_loss = criterion(denorm_output, denorm_label)

                total_loss += denorm_loss.item()
                loss.backward()
                optimizer.step()
                psnr_total += PSNR(denorm_loss.item(), Ma - mi)

            train_loss.append((total_loss/num_t_samples)**0.5)
            print("===> Epoch[{}]: , loss: {:.10f}, PSNR: {:.10f}".format(epoch, (total_loss / num_t_samples)**0.5,
                                                                          psnr_total / num_t_samples))
#            if epoch  == 10:
#                torch.save(model.state_dict(), os.path.join(opn.outputs_dir, 'best_epoch_{}.pth'.format(epoch)))

            model.eval()
            total_val_loss = 0
            for iteration, batch in enumerate(Val_data):
                val_input = torch.stack([batch[0]])
                val_label = torch.stack([batch[1]])
                [mi, Ma] = minmax_v[iteration]

                # 모델을 불러올때, grad 계산을 생략해서 메모리 효율을 높인다.
                with torch.no_grad():
                    pred = model(val_input)

                denorm_pred = pred  * (Ma - mi) + mi
                denorm_val_label = val_label  * (Ma - mi) + mi
                denorm_val_loss = criterion(denorm_pred, denorm_val_label)
                total_val_loss += denorm_val_loss.item()

            eval_loss.append((total_val_loss/num_v_samples)**0.5)
            print("             Validation loss: {:.10f}".format((total_val_loss/ num_v_samples)**0.5))

    ##rmse of epoch
    train_loss = []
    eval_loss = []
    for epoch_ in range(epoch, opn.numEpochs + 1):
        train(Train_data, optimizer, model, criterion, epoch_, train_loss, eval_loss)

        if epoch_ == opn.numEpochs:
            plt.figure()
            plt.plot(train_loss, label = 'train')
            plt.plot(eval_loss, label = 'validation')
            plt.show()




# if locally activated
if __name__ == '__main__':
    print('locally run')
    main()