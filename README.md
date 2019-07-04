## TWBA-Nets: Deep neural networks with ternary weights {-1,0,+1} and binary activations {0,1} 
***

**This code can be used as a reference for the paper: "Deep Spiking Convolutional Neural Networks for Programmable Neuro-synaptic System"**
***

### Citation:
Chenglong Zou, Xinan Wang, Boxing Xu, Yisong Kuang, Xiaoxin Cui. Deep Spiking Convolutional Neural Networks for Programmable Neuro-synaptic System, The 13th IEEE International Conference on ASIC (ASICON 2019).


### Features:
- This example is the experiment of BWBA-Nets/TWBA-Nets for "Model-2"  based on CIFAR-10 dataset in above paper. You can manually decay the learning rate from original 0.01 by 10× every 200 epochs while restore from corresponding checkpoint by setting flag "resume = True", the final test accuracy will be about 86.50% for BWBA-Nets, 88.19% for TWBA-Nets, respectively.

- You can furtherly count the spikes generated in TWBA-Nets running the "cifar10_twba_spikes.py". 

- It should be noted that spikes counting does not consider the first transferring convolutional layer and last 1*1 convolutional layer, because these layers are computed off chip, and batchsize is 200.

This work relies on Tensorflow==1.5.0 and Tensorlayer==1.8.4, we only modify a little for our design. 

### Requirements:<br>
1. Python 3.5<br>
2. Tensorflow 1.5.0 for cpu or gpu<br>
3. Tensorlayer 1.8.4 (An open community to promote AI technology).<br> 
(seeing https://github.com/tensorlayer for more information)<br>
4. Please replace the script "binary.py" (seeing in your_python_path\lib\site-packages\tensorlayer\layers\binary.py) with provided "binary.py".



### File overview:

BWBA - the folder for BWBA example.  
BWBA/cifar10_bwba_example.py - the BWBA example script on CIFAR10 dataset.
TWBA - the folder for TWBA example.
TWBA/cifar10_twba_example.py - the TWBA training example script on CIFAR10 dataset.
TWBA/cifar10_twba_spikes.py - the test script for counting spikes on CIFAR10 dataset.
binary.py - the provided quantization script for replacing original script.
README.md - this readme file.


### Usage:
- After installing the package Tensorflow and Tensorlayer, you should manually replace the script "binary.py" (path: your_python_path\lib\site-packages\tensorlayer\layers\binary.py) with provided "binary.py.

- Then you can directly run the script "TWBA/cifar10_twba_example.py" or "BWBA/cifar10_bwba_example.py" for training.

- Finally, if you want to count spikes generated in the model described in "cifar10_twba_example.py", you can just run the script "cifar10_twba_spikes.py" for test.


### More question:
Please feel free to reach out here or email: 1801111301@pku.edu.cn if you have any questions or difficulties. I'm happy to help guide you.


