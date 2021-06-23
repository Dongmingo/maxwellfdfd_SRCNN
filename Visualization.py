import torch
import matplotlib.pyplot as plt

#[#samples, width, height]로 구성된 3차원 matrix를 rows * cols 의 subplot으로 출력해 보여준다.
def Visualize_samples(mat, rows, cols, epoch):
    fig = plt.figure(figsize=(24,13))
    axes = []

    for i in range(cols*rows):
        temp = torch.tensor(mat[i])
        axes.append(fig.add_subplot(rows, cols, i+1))
        subplot_title = ('Sample'+str(i+1))
        axes[-1].set_title(subplot_title)
        #plt.imshow(temp, cmap = 'hot')
        plt.imshow(temp, cmap = 'hot', vmin = 0, vmax = 2.6)


    fig.tight_layout()
    fig.canvas.manager.set_window_title('epoch_'+str(epoch))
    plt.show()