import numpy as np
import argparse
import torch
import torch.nn as nn
from model_SRCNN import Net
from Data_augment import h5_DataAug
from Visualization import Visualize_samples


parser = argparse.ArgumentParser()
parser.add_argument('--weights-file', type=str, default= 'saved_state/best_epoch_500.pth')
parser.add_argument('--testxfile', type=str, default='20samples_2/testx.h5')
parser.add_argument('--testyfile', type=str, default='20samples_2/testy.h5')
parser.add_argument('--scale', type=int, default = 2)
parser.add_argument('--mode', default='client')
parser.add_argument('--port', default=61401)

def main():
    global opn, optimizer
    opn = parser.parse_args()
    print(opn)

    print('===> Loading Dataset')
    input, label, minmax = h5_DataAug(opn.testxfile, opn.testyfile, scale=opn.scale)
    Test_data = torch.transpose(torch.stack([input, label]), 0, 1)
    num_t_samples = list(Test_data.size())[0]

    def PSNR(loss, minmax):
        psnr = 10 * np.log10((minmax ** 2) / loss)
        return psnr

    print('===> Setting model')
    model = Net()
    criterion = nn.MSELoss(reduction='mean')
    model.load_state_dict(torch.load(opn.weights_file))
    vis_input = []
    vis_pred = []
    vis_label = []

    def test(Test_data, model, criterion):
        model.eval()
        total_test_loss = 0
        psnr_total = 0
        sample_loss = []
        for iteration, batch in enumerate(Test_data):
            test_input = torch.stack([batch[0]])
            test_label = torch.stack([batch[1]])
            [mi, Ma] = minmax[iteration]

            with torch.no_grad():
                pred = model(test_input)

            denorm_input = test_input * (Ma - mi) + mi
            denorm_pred = pred * (Ma - mi) + mi
            denorm_test_label = test_label * (Ma - mi) + mi
            denorm_test_loss = criterion(denorm_pred, denorm_test_label)
            sample_loss.append(denorm_test_loss.item())
            total_test_loss += denorm_test_loss.item()
            psnr_sample = PSNR(denorm_test_loss.item(),Ma-mi)
            if iteration == 8:
                input_loss = criterion(denorm_input, denorm_test_label).item()
                sample_psnr = PSNR(criterion(denorm_input, denorm_test_label), Ma-mi)
            psnr_total += psnr_sample

            vis_input.append(denorm_input[0][0])
            vis_label.append(denorm_test_label[0][0])
            vis_pred.append(denorm_pred[0][0])

        print("\t sample number : Spline Cubic input")
        print("===> sample_rmse:, {:.10f}, sample-PSNR : {:.10f}".format(input_loss, sample_psnr))

        #for i in range(num_t_samples):
        #    print("{:.10f}".format(sample_loss.pop()**0.5))

        #print("\n===> RMSE: {:.10f} PSNR: {:.10f}".format((total_test_loss / num_t_samples) ** 0.5,
        #                                                              psnr_total / num_t_samples))
        Visualize_samples([vis_input[8]], 1, 1, 'input')
        #Visualize_samples([vis_pred[8]], 1, 1, 'pred')
        #Visualize_samples(vis_label, 2, 10, 'label')

    print('===> test_data {:3d}samples'.format(num_t_samples))
    test(Test_data, model, criterion)

if __name__ == '__main__':
    print('locally run')
    main()
