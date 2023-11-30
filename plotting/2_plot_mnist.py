import pickle
import sys
sys.path.append('..')  # Adds the parent directory to the Python path1
import matplotlib.pyplot as plt
from my_utils.utils_reading_disks import get_dict_from_yaml
from my_utils.utils_dataloader import get_ds_mnist
import numpy as np

def count_classes(dataset, indices, configs, idx_client):
    global class_counts
    for idx in indices:
        target = dataset.targets[idx].item()
        class_counts[idx_client][target] += 1
    

path_config = '../configs/4_mnist_sra_fl_non_iid.yaml'
configs = get_dict_from_yaml(path=path_config)
print(configs)
class_counts = np.zeros((configs['num_clients'], configs['num_class']), dtype=int)

ds_train, ds_test = get_ds_mnist()

folder_idx = '../idx_'+configs['exp_name']
with open(folder_idx+'/idxs_'+str(configs['degree_non_iid'])+'.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    f.close()

for idx_client in range(configs['num_clients']):
    idxs_cur = np.array(data_dict[idx_client]).tolist()
    # print(type(idxs_cur), idxs_cur); import sys; sys.exit()
    count_classes(dataset=ds_train, indices=idxs_cur, configs=configs, idx_client=idx_client)

print(class_counts, np.sum(class_counts))
 

with open('../idx_'+configs['exp_name']+'_accs_'+str(configs['degree_non_iid'])+'.pkl', 'rb') as f:
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