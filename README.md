# Datasets
### Samples
- 298 samples of 300X300 images with labels
- 5040 samples of 300x300 images with labels
- 2669 samples of 300x300 images with labels

### Labels
- **0** -> Defective  
- **1** -> Non Defective


# File Description
`slenet.py` Running the file gives inference details on the raw dataset  
`slenet.ipynb` Contains all the experimentation done while creating slenet  
`slenet.th` The Slenet Model  
`salexnet.py` Running the file gives inference details on the raw dataset  
`salexnet.ipynb` Contains all the experimentation done while creating salexnet  
`salexnet.th` The Salexnet Model  
`epochs` Contains tensorboard visualization runs for number of epochs  
`learning_rates` Contains tensorboard visualization runs for learning rates
`requirements.txt` list of modules required to run the files' and notebooks' code  
  
  # Installing Modules  
  ```shell
  $ pip3 install -r requirements.txt
  ```  
  # Running Tensorboard Visualizations
  ```shell
  $ tensorboard --logdir=learning_rates
  ```
  should give
  ```shell
  TensorBoard 1.14.0 at http://shouvik-Predator-PH315-51:6006/ (Press CTRL+C to quit)
  ```
  Open the URL in a browser
  
  # Inference
  ```shell
  $ python3 slenet.py
  ```
  should give
  ```shell
  RawAccuracy:0.6234544773323342
  Raw Dataset Confusion Matrix
  [[ 147    2]
  [1003 1517]]
  Raw Dataset F1-Score: 0.7218757844113998
  0 misclassification: 1.342281879194631 %
  1 misclassification: 39.8015873015873 %
  ```
