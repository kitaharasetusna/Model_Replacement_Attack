# Model Replacement Attack

# About this project
In this repo, I reimplemented 'Adversarial Weight Attack Paradigm' based model replacement attacks to learn backdoor attacks on a fine-grained scale.

To do the benchmakr, I also plan to reimplemente some other attacks and defense on both centrailized and the FL setting.

This guidance provides some simple cases.
For more complicated case, please refer to the `./docs` folder.

# TODO List
- [x] channel based subnet replacement attack (CVPR 2022)
- [x] non-IID dataloaders
- [ ] FL setting attack and defense framework
- [ ] Add untargeted attack
- [ ] layer-wise subnet replacement attack
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

## layer-wise SRA (FL setting)

`remark`: it could be interpreted as an adaptive attack (adapitive subnet along the layers)