# Experiment-of-Attention-Modules-on-WaferMap-Classification

> This project mainly focuses on discovering the infleunces brought by different attention modules on residual networks. Those attention modules derive from **CBAM**, **DANet** and **PAN**. Also, each model is tested on **WM811K** dataset for futher evaluation. In addition the numerical performances, such as accracy and f1-score, we also adopt GradCAM and GradCAM++ for visualization. Noted that all the class activation maps are captured before entering the average pooling layer consecutively connected with classifier. Eventually, in this experiment we apply pre-trained resnet34 as backbone to compare the perofrmances only.

## Files Intorduction
### myFunc.py
* All the training, testing, evaluation functions are contained in thise file. Futhermore, the functions that help showing GradCAM, GradCAM++ or fixing the random seeds are available in this .py file.
### models.ipynb
* Contain basic models and modified models, which the basic models are referred to github source code shown in the references.
### results.pdf
* Provide the evalutaions of different modified model including Recall, F1-score, F1-macor, Accuracy, GradCAM++.

## Basic Models
### 1. DANet
![DANet](https://github.com/Paddyyhqhi/Experiment-of-Attention-Modules-on-WaferMap-Classification/assets/126771856/69fef376-6afd-40f0-94bf-836d020c6f08)
### 2. CBAM
![CBAM](https://github.com/Paddyyhqhi/Experiment-of-Attention-Modules-on-WaferMap-Classification/assets/126771856/65f9c4ea-011e-4669-b392-92e1450d4660)
### 3. PAN
![PAN](https://github.com/Paddyyhqhi/Experiment-of-Attention-Modules-on-WaferMap-Classification/assets/126771856/b5702bcf-207b-4fb6-b6ab-cb9a46f3c1af)

## Modifications
### 1. CBAM_DA
* Replcae the squeeze and extension parts of spatial, channel attention module into position, channel attention modules in DANet.
![CBAM_DA](https://github.com/Paddyyhqhi/Experiment-of-Attention-Modules-on-WaferMap-Classification/assets/126771856/91edb7ea-d673-42b1-b382-c97391a38b46)
### 2. PAN_DA
* Integrate the feautre map of PAN and DA by model fusion.
![PAN_DA](https://github.com/Paddyyhqhi/Experiment-of-Attention-Modules-on-WaferMap-Classification/assets/126771856/4891e640-af64-4ee2-9ccc-8742dcb9f1f9)
### 3. PAN_Inverse
* Modify the GAU into GAD to merge feature maps of different level but from low-level to higher one.
![PAN_inv](https://github.com/Paddyyhqhi/Experiment-of-Attention-Modules-on-WaferMap-Classification/assets/126771856/38218c75-7b8f-4cca-8393-1627c642a09b)
### 4. PAN_DualInverse
* Add another branch of Global Attention Donwsmaple Spatial to consider not only the aspect of channel integration.
![PAN_DualInverse](https://github.com/Paddyyhqhi/Experiment-of-Attention-Modules-on-WaferMap-Classification/assets/126771856/9e52d084-644c-4ddb-bea1-47a67d825355)

## References
1. DANet : https://github.com/junfu1115/DANet
2. PAN : https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch
3. CBAM : https://github.com/luuuyi/CBAM.PyTorch







