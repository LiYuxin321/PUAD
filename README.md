# Code for the paper MetaAD: A Prototype-oriented Meta Anomaly Detection Framework for Multivariate Time Series

## To run with local environment

Install dependencies using python 3.8+ : 

```
pip install -r requirements.txt 
```

Run the model: 
```
python runner10.py --test_train_step2 10
```

test_train_step2 is the number of data used for meta-training.
## Data

### SMD：
Download the processed data form Google cloud to './SMD/'.
[SMD Data](https://drive.google.com/file/d/1rv6NMgb2F3CubasCiUayNnrSQXNvNPGC/view?usp=sharing)。


### MSL：
Run the data process code in the './MSL/'.
```
python ./MSL/data_press_1.py
```





