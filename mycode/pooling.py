from typing import Optional

import torch


def concat_pool(x, batch, size: Optional[int]=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = int(batch.max().item() + 1) if size is None else size
    global_emd = torch.tensor([], device=device)
    # global_emd = global_emd.to(device)
    itertime = 0
    for i in range(size):
        tmp = torch.tensor([], device=device)
        # tmp = tmp.to(device)
        count = 0
        for j in range(itertime, len(x)):
            if batch[j].item() == i:
                count += 1
                tmp = torch.cat((tmp, x[j]), dim=0)
            else:
                itertime = j
                break

        tmp = torch.cat((tmp.view(1, -1), torch.zeros(1, (51-count)*x.size(1)).to(device)), dim=1)
        global_emd = torch.cat((global_emd, tmp.view(1, -1)), dim=0)
    return global_emd
