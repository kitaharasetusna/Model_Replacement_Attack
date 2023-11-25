# Model_Replacement_Attack
In this repo, I reimplemented 'Adversarial Weight Attack Paradigm' based model replacement attacks


## centrailized  backdoor attack
An example for [Towards Practical Deployment-Stage Backdoor Attack on Deep Neural Networks](https://arxiv.org/abs/2111.12965)

run this script:
```
cd .\examples\
```
```
python ./cifar_10_sra.py
```
To train from scratch, you should change the configuration of 2 `training` values in the dictionary config to True first.