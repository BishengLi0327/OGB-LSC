import torch
import matplotlib.pyplot as plt

# dropout ratio：0.1
val_accu_gcn = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                          'adjust/GCN/dropout1/CHECKPOINT_DIR_GCN/checkpoint.pt')['best_val_mae']
val_accu_gcnv = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                           'adjust/GCNV/dropout1/CHECKPOINT_DIR_GCNV/checkpoint.pt')['best_val_mae']
val_accu_gin = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                          'adjust/GIN/dropout1/CHECKPOINT_DIR_GIN/checkpoint.pt')['best_val_mae']
val_accu_ginv = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                           'adjust/GINV/dropout1/CHECKPOINT_DIR_GINV/checkpoint.pt')['best_val_mae']

# dropout ratio：0.2
val_accu_gcn2 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                           'adjust/GCN/dropout2/CHECKPOINT_DIR_GCN/checkpoint.pt')['best_val_mae']
val_accu_gcnv2 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                            'adjust/GCNV/dropout2/CHECKPOINT_DIR_GCNV/checkpoint.pt')['best_val_mae']
val_accu_gin2 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                           'adjust/GIN/dropout2/CHECKPOINT_DIR_GIN/checkpoint.pt')['best_val_mae']
val_accu_ginv2 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                            'adjust/GINV/dropout2/CHECKPOINT_DIR_GINV/checkpoint.pt')['best_val_mae']

# dropout ratio：0.5
val_accu_gcn5 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                           'adjust/GCN/dropout5/CHECKPOINT_DIR_GCN/checkpoint.pt')['best_val_mae']
val_accu_gcnv5 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                            'adjust/GCNV/dropout5/CHECKPOINT_DIR_GCNV/checkpoint.pt')['best_val_mae']
val_accu_gin5 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                           'adjust/GIN/dropout5/CHECKPOINT_DIR_GIN/checkpoint.pt')['best_val_mae']
val_accu_ginv5 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                            'adjust/GINV/dropout5/CHECKPOINT_DIR_GINV/checkpoint.pt')['best_val_mae']


# dropout ratio：0.7
val_accu_gcn7 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                           'adjust/GCN/dropout7/CHECKPOINT_DIR_GCN/checkpoint.pt')['best_val_mae']
val_accu_gcnv7 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                            'adjust/GCNV/dropout7/CHECKPOINT_DIR_GCNV/checkpoint.pt')['best_val_mae']
val_accu_gin7 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                           'adjust/GIN/dropout7/CHECKPOINT_DIR_GIN/checkpoint.pt')['best_val_mae']
val_accu_ginv7 = torch.load('/root/libisheng/OGB-LSC/demo_code/'
                            'adjust/GINV/dropout7/CHECKPOINT_DIR_GINV/checkpoint.pt')['best_val_mae']

gcn = [val_accu_gcn, val_accu_gcn2, val_accu_gcnv5, val_accu_gcn7]
gcnv = [val_accu_gcnv, val_accu_gcnv2, val_accu_gcnv5, val_accu_gcnv7]
gin = [val_accu_gin, val_accu_gin2, val_accu_gin5, val_accu_gin7]
ginv = [val_accu_ginv, val_accu_ginv2, val_accu_ginv5, val_accu_ginv7]

plt.figure()
x = [0.1, 0.2, 0.5, 0.7]
plt.plot(x, gcn, color='green', label='gcn')
plt.plot(x, gcnv, color='red', label='gcnv')
plt.plot(x, gin, color='skyblue', label='gin')
plt.plot(x, ginv, color='blue', label='ginv')
plt.xlabel('dropout ratio')
plt.ylabel('mae')
plt.legend()
plt.show()
