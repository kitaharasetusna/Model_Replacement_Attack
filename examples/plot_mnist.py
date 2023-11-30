import pickle
import sys
sys.path.append('..')  # Adds the parent directory to the Python path1
import matplotlib.pyplot as plt
from my_utils.utils_reading_disks import get_dict_from_yaml



path_config = '../configs/4_mnist_sra_fl_non_iid.yaml'
configs = get_dict_from_yaml(path=path_config)
print(configs)

folder_idx = '../idx_'+configs['exp_name']
with open(folder_idx+'/idxs_'+str(configs['degree_non_iid'])+'.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    f.close()

with open('../idx_'+configs['exp_name']+'_accs.pkl', 'rb') as f:
    test_accuracy = pickle.load(f) 
    f.close()
print(test_accuracy)

train_accs =[]
test_accs = []
for tarin_acc, test_acc in test_accuracy:
    test_accs.append(test_acc)
print(train_accs)
print(test_accs)


t = range(0, len(test_accs)*configs['time_step'], configs['time_step'])
plt.figure(figsize=(5, 4))
plt.plot(t, test_accs)
plt.xlabel('epoch')
plt.ylabel('accuracy (%)')
plt.show()