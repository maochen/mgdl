import sys
from subprocess import call

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet101

if __name__ == "__main__":

    if not torch.cuda.is_available():
        raise ValueError("No CUDA compatible device found.")

    batchsize = int(sys.argv[1]) if len(sys.argv) > 1 else 200  # 200 for 4 V100 16G GPU
    print("batch size: " + str(batchsize))

    nclasses = 30
    size = 10000
    x = np.random.rand(size, 3, 256, 256)
    y = np.random.randint(low=0, high=nclasses, size=size)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()

    dataset = TensorDataset(x, y)
    dl = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=False, num_workers=10)

    net = DataParallel(resnet101()).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1.2)

    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    ep = 0
    while True:
        ep += 1
        optimizer.zero_grad()
        correct = 0

        for curr_x, curr_y in tqdm.tqdm(dl, total=len(dl), postfix="EP " + str(ep)):
            curr_x = curr_x.cuda()
            curr_y = curr_y.cuda()

            pred = net(curr_x)
            loss = criterion(pred, curr_y)
            loss.backward()
            optimizer.step()

            pred_max_index = pred.max(dim=1)[1]
            correct += (pred_max_index.eq(curr_y.long())).sum().item()

        # if ep % 1 == 0:
        #     acc = correct / len(dataset)
        #     print(f"EP: {ep} acc: {acc}")
