# Model Replacement Attack
This is a federated learnig repo for the research purpose.
It includes Fl framwork for MNIST(non-iid/iid), CIFAR-10(non-iid/iid)

This repo is aimed at making benchmarks for before/after attacks.
# About this project


In this repo, I reimplemented 'Adversarial Weight Attack Paradigm' based model replacement attacks to learn backdoor attacks on a fine-grained scale.

To do the benchmakr, I also plan to reimplemente some other attacks and defense on both centrailized and the FL setting.

This guidance provides some simple cases.
For more complicated case, please refer to the `./docs` folder.

# TODO List
- [x] channel based subnet replacement attack (CVPR 2022)
- [x] non-IID dataloaders (MNIST)
- [x] non-IID dataloaders (CIFAR-10)
- [ ] FL setting attack and defense framework
- [ ] Add untargeted attack
- [ ] layer-wise subnet replacement attack
    - [ ] add check or not to tell one label or all labels
- [ ] Filp defense (an inversion-based defence)


## Centrailized  Backdoor Attack


An example for [Towards Practical Deployment-Stage Backdoor Attack on Deep Neural Networks](https://arxiv.org/abs/2111.12965)

run this script:
```
cd .\examples\
```
```
python ./1_cifar_10_sra.py
```
To train from scratch, you should change the configuration of 2 `training` values in the dictionary config to True first.


## Federated Learning Setting
An example for Non-IID datasets and dataloaders for clients and server from the paper [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335)

to run this code

use
```
cd .\examples\
```
```
python .\2_cifar_10_non_iid.py
```

## [Federated Learning Setting MNIST(non-iid)](./examples/4_MNIST_sra_fl_non_iid.py)
|  Number clients | C (fraction/round) | loal epoch | optimizer| learning rate|
| :----------- | :------------: | ------------: | ------------: |------------: |
| 64        |   0.25        |   2       | Adam|1e-3


|  Non-iid | Model | test acc | epochs|
| :----------- | :------------: | ------------: | ------------: |
| IID       |    CNN MNIST        |    81.38       | 300|
| 0.1       |    CNN MNIST        |    58.57       | 300|

use 
```
cd .\examples\
```
```
4_MNIST_sra_fl_non_iid.py
```
## [Federated Learning Setting CIFAR-10(non-iid)](./examples/5_cifar10_fl_non_iid.py)
|  Number clients | C (fraction/round) | loal epoch | optimizer| learning rate|
| :----------- | :------------: | ------------: | ------------: |------------: |
| 100        |   0.1        |   5       | Adam|1e-4


|  Non-iid | Model | test acc | epochs|
| :----------- | :------------: | ------------: | ------------: |
| 0.5        |    ResNet50        |    63.4      | 300|
| IID        |    ResNet50        |    72.3      | 300|
| IID        |    ResNet18        |    73.67      | 300|
```
cd .\examples\
```
```
python 5_cifar10_fl_non_iid.py
```


## [Federated Learning Setting CIFAR-100(non-iid)](./examples/7_cifar100_resnet18_non_iid.py)
|  Number clients | C (fraction/round) | loal epoch | optimizer| learning rate|
| :----------- | :------------: | ------------: | ------------: |------------: |
| 100        |   0.1        |   5       | Adam|1e-4


|  Non-iid | Model | test acc | epochs|
| :----------- | :------------: | ------------: | ------------: |
| 0.3        |    ResNet50        |         | 300|
| IID        |    ResNet50        |    34.1   | 300|
```
cd .\examples\
```
```
python 7_cifar100_resnet18_non_iid.py 
```

## layer-wise SRA (FL setting)

`remark`: it could be interpreted as an adaptive attack (adapitive subnet along the layers)