import pickle
import sys
sys.path.append('..')  # Adds the parent directory to the Python path1
import matplotlib.pyplot as plt
from my_utils.utils_reading_disks import get_dict_from_yaml



path_config = '../configs/5_cifar_10_sra_fl_non_iid.yaml'
configs = get_dict_from_yaml(path=path_config)
print(configs)

folder_idx = '../idx_'+configs['exp_name']
with open(folder_idx+'/idxs.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    f.close()

with open('../idx_'+configs['exp_name']+'_accs.pkl', 'rb') as f:
    test_accuracy = pickle.load(f) 
    f.close()
print(test_accuracy)
test_accuracy = list(test_accuracy)
train_accs =[]
test_accs = []
for (train_acc, test_acc) in test_accuracy:
    train_accs.append(train_acc)
    test_accs.append(test_acc)
print(train_accs)
print(test_accs)


# plt.figure((10, 10))
# plt.plot(test_accuracy)
# plt.show()