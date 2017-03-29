## PyTorch Implementation of the Coupled Generative Adversarial Networks (CoGAN)
---
### General
This is a PyTorch implementation of the Coupled Generative Adversarial Netowork algorithm. For more details please refer to our [NIPS paper](https://papers.nips.cc/paper/6544-coupled-generative-adversarial-networks.pdf) or our [arXiv paper](https://arxiv.org/abs/1606.07536). Please cite the NIPS paper in your publications if you find the source code useful to your research.
---
### Installation
In your python package, install [pytorch](https://github.com/pytorch/pytorch) and [torchvision](https://github.com/pytorch/vision). You also need yaml, Python Opencv and Google Logging to run the code.
---
### Usage
---
#### Simple example

Train the CoGAN network to learn to generate digit images and the corresponding edges images of the digits images without the need of corresponding images in the two domains in the training dataset.
```
cd src;
python train_cogan_mnistedge.py --config ../exps/mnistedge_cogan.yaml;
```
..After 5000 iterations, you will see the generation results in outputs/mnistedges_cogan/ and they should look like.
..![](https://github.com/mingyuliutw/CoGAN_PyTorch/blob/master/outputs/mnistedge_cogan/mnistedge_cogan_gen_00005000.jpg)
---
#### Domain adaptation using all training images

Train the CoGAN network to unsupervisedly adapt a digit classifier from the MNIST domain to the USPS domain by using all the images in the training sets. Use 60000 images from the MNIST training set when unsupervisedly adapting from MNIST to USPS. Use 7438 images from the USPS training set when unsupervisedly adapting from USPS to MNIST. 
```
cd src;
python train_cogan_mnist2usps.py --config ../exps/mnist2usps_full_cogan.yaml;
python train_cogan_usps2mnist.py --config ../exps/usps2mnist_full_cogan.yaml;
```
You will see the accuracy of the adapted classifier in the test set in the target domain in the log file. The best accuracy in your log files should be something like

| Setting | MNIST to USPS | USPS to MNIST |
| ------- |:-------------:|:-------------:|
| CoGAN   | 0.95XX        | 0.93XX        |

---
#### Domain adaptation using a subset of training images

Train the CoGAN network to unsupervisedly adapt a digit classifier from the MNIST domain to the USPS domain by using subsets of the training sets. Use 2000 images from the MNIST training set when unsupervisedly adapting from MNIST to USPS. Use 1800 images from the USPS training set when unsupervisedly adapting from USPS to MNIST. 
```
cd src;
python train_cogan_mnist2usps.py --config ../exps/mnist2usps_small_cogan.yaml;
python train_cogan_usps2mnist.py --config ../exps/usps2mnist_small_cogan.yaml;
```
You will see the accuracy of the adapted classifier in the test set in the target domain in the log file. The best accuracy in your log files should be something like

| Setting | MNIST to USPS | USPS to MNIST |
| ------- |:-------------:|:-------------:|
| CoGAN   | 0.94XX        | 0.92XX        |
