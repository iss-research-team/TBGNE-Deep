# TBGNE-Deep
An AMT adoption prediction model based on neighbor in multilayer networks, and enriched by deep learning. 
## Prerequisites 
Python 3  
PyTorch  
sklearn  
## Getting Started 
Clone this repo.  
```
git clone https://github.com/iss-research-team/TBGNE-Deep/  
cd TBGNE-Deep  
```
## Dataset  
Due to the relevant regulations of the data provider, these datasets are sampled from the original datasets. 
T(echnology)B(usiness)G(eography) data, from Ningbo, Zhejiang, China, contains 36226 nodes and 94103 edges.  
If complete data and results are required, please contact us by gnxu@bupt.edu.cn.  
## Training  
Training on the existing datasets  
Users can use the sample data in /data, or they can make their own dataset.  
Representation of node in multilayer networks can be calculated by running /src/run_pytorch.py  
Predict model can be trained by running /src/predict_model.py  
And users can get assessment of predict model in /src/predict_model.py  
## Predict model
Predict model is saved after trainning by running /src/predict_model.py, which can be used in other AMT adoption prediction tasks. 
 
