# Experiment-of-Attention-Modules-on-WaferMap-Classification
## Objective
  This project maily focuses on discovering the infleunces brought by different attention modules on residual networks. Those attention modules derive from **CBAM**, **DANet** and **FPA**.
  Also, each model is tested on **WM811K** dataset for futher evaluation. In addition the numerical performances, such as accracy and f1-score, we also adopt GradCAM and GradCAM++ for visualization. Noted that all the class activation maps are captured before entering the average pooling layer consecutively connected with classifier.
## Files Intorduction
  ### myFunc.py
  All the training, testing, evaluation functions are contained in thise file. Futhermore, the function that helps showing GradCAM, GradCAM++ are available in this .py file.
  ### models.ipynb
  Filled in the future.
  ### results.ipynb
  Provide the evalutaions of different modified models.
## Models
This part provide the graphs of models for people to understading the modifications more intuitively.

