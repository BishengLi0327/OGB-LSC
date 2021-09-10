import torch
import numpy as np

# for validation
val_acc_gcn = torch.load('/root/libisheng/OGB-LSC/demo_code/CHECKPOINT_DIR_GCN/checkpoint.pt')['best_val_mae']
val_acc_gcnv = torch.load('/root/libisheng/OGB-LSC/demo_code/CHECKPOINT_DIR_GCNV/checkpoint.pt')['best_val_mae']
val_acc_gin = torch.load('/root/libisheng/OGB-LSC/demo_code/CHECKPOINT_DIR_GIN/checkpoint.pt')['best_val_mae']
val_acc_ginv = torch.load('/root/libisheng/OGB-LSC/demo_code/CHECKPOINT_DIR_GINV/checkpoint.pt')['best_val_mae']
val_acc_mlp = torch.load('/root/libisheng/OGB-LSC/demo_code/CHECKPOINT_DIR_MLP/checkpoint.pt')['best_val_mae']

# for test
# test_acc_gcn = torch.load('/root/libisheng/OGB-LSC/demo_code/CHECKPOINT_DIR_GCN/checkpoint_test.pt')['test_mae']
# test_acc_gcnv = torch.load('/root/libisheng/OGB-LSC/demo_code/CHECKPOINT_DIR_GCNV/checkpoint_test.pt')['test_mae']
# test_acc_gin = torch.load('/root/libisheng/OGB-LSC/demo_code/CHECKPOINT_DIR_GIN/checkpoint_test.pt')['test_mae']
# test_acc_ginv = torch.load('/root/libisheng/OGB-LSC/demo_code/CHECKPOINT_DIR_GINV/checkpoint_test.pt')['test_mae']

print('model: gin \t \t gin-virtual gcn \t \t gcn-virtual mlp')
print('mae:   %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (val_acc_gin, val_acc_ginv, val_acc_gcn, val_acc_gcnv, val_acc_mlp))
# print('taccu: %.4f \t %.4f \t %.4f \t %.4f' % (test_acc_gin, test_acc_ginv, test_acc_gcn, test_acc_gcnv))
