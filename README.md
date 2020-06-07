

#Data

Please download [data][https://static.aminer.cn/misc/na-data-kdd18.zip] here. Unzip the file and put the data directory into project directory.



# RUN
```
python3 scripts/preprocessing.py

# global model
python3 HeterogeneousGraph/gen_train_data.py
python3 HeterogeneousGraph/global_model.py
python3 HeterogeneousGraph/prepare_local_data.py

# train
python3 local/gae/train.py

```











