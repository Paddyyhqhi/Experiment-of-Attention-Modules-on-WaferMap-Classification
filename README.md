# Experiment-of-Attention-Modules-on-WaferMap-Classification
## Objective
This project maily focuses on discovering the infleunces brought by different attention modules on residual networks. Those attention modules derive from **CBAM**, **DANet** and **FPA**.
Also, each model is tested on **WM811K** dataset for futher evaluation. In addition the numerical performances, such as accracy and f1-score, we also adopt GradCAM and GradCAM++ for visualization. Noted that all the class activation maps are captured before entering the average pooling layer consecutively connected with classifier. Eventually, in this experiment we apply pre-trained resnet34 as backbone to compare the perofrmances only.
## Files Intorduction
### myFunc.py
All the training, testing, evaluation functions are contained in thise file. Futhermore, the functions that help showing GradCAM, GradCAM++ or fixing the random seeds are available in this .py file.
### models.ipynb
Filled in the future.
### results.ipynb
Provide the evalutaions of different modified models.
## Models
This part provide the graphs of models for more intuitively understading the modifications.
![DANet](https://github.com/Paddyyhqhi/Experiment-of-Attention-Modules-on-WaferMap-Classification/assets/126771856/b2eb2fd9-3cc7-4859-b7c3-648eb2cbab5b)

