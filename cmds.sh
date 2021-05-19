#python3 eval_tune.py freeze_prepro=1 model=tabnet dts=clinical pp=7 nt=1000 --multirun
#python3 data_preprocessing.py --remove_precipitals 0
python3 eval_tune.py freeze_prepro=1 model=xgb,log dts=clinical,sparse_img,clinical_sparse_img pp=15 --multirun
python3 data_preprocessing.py --remove_precipitals 0
python3 eval_tune.py freeze_prepro=1 model=log,xgb dts=clinical,sparse_img,clinical_sparse_img pp=15 --multirun
python3 data_preprocessing.py --remove_precipitals 1
python3 eval_tune.py freeze_prepro=1 model=torch,mlp dts=clinical_blood_sparse_img pp=4 nt=1000 --multirun
python3 eval_tune.py freeze_prepro=1 model=torch,mlp,tabnet dts=clinical_blood_sparse_img pp=4 nt=5000 --multirun
