# Code for the paper MetaAD: A Prototype-oriented Meta Anomaly Detection Framework for Multivariate Time Series

[Link (ICML2023)](https://openreview.net/forum?id=3vO4lS6PuF&referrer=%5Bthe%20profile%20of%20Yuxin%20Li%5D(%2Fprofile%3Fid%3D~Yuxin_Li3))
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
[SMD Data](https://drive.google.com/drive/folders/1H7J67sKOd1ogGa-IzgNw99WyPYJxyuYI?usp=share_link)。


### MSL：
Run the data process code in the './MSL/'.
```
python ./MSL/data_press_1.py
```





