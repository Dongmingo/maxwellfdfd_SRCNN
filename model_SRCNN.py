import torch
import torch.nn as nn

# input, output : 4dim dataset [# of inputs, depth of input, Height, Width]

class Net(nn.Module):

    def __init__(self, input_depth = 1, c1 = 64, c2 = 32, c3 = 1, k1 = 9, k2 = 1, k3 = 5):
        super().__init__()
        # kernel_size : k1 x k1 size #c1 conv filters
        self.conv1 = nn.Conv2d(in_channels=input_depth, out_channels=c1, kernel_size=k1, padding=(k1-1)//2)
        self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=k2, padding=(k2-1)//2)
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=k3, padding=(k3-1)//2)

        #initial weight
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

        #bias 초기값
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        return x