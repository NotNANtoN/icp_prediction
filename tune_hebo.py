import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import sklearn
import matplotlib.pyplot as plt

from icp_pred.tune_utils import obj


def obj_hebo(*args, **kwargs):
    return 1 - np.array([obj(*args, **kwargs)])
    

def get_hebo_space(model_type):
    if model_type in ["rnn", "gpt", "mlp"]:
        space = DesignSpace().parse([{'name': 'lr', 'type' : 'num', 'lb' : 0.00001, 'ub' : 0.1},
                                    {'name': 'bs', 'type' : 'int', 'lb' : 2, 'ub' : 5},  # 2 ** bs
                                    
                                    {'name': 'fill_type', 'type' : 'cat', 'categories' : ['median', 'none']},
                                    
                                    #{'name': 'min_len', 'type' : 'int', 'lb': 2, 'ub':128},
                                    #{'name': 'max_len', 'type' : 'int', 'lb': 64, 'ub':512},
                                    
                                    #{'name': 'train_noise_std', 'type' : 'int', 'lb' : 0, 'ub' : 2},
                                    {'name': 'weight_decay', 'type' : 'int', 'lb' : 0, 'ub' : 4},
                                    {'name': 'grad_clip_val', 'type' : 'int', 'lb' : 0, 'ub' : 5},
                                    #{'name': 'norm_method', 'type' : 'cat', 'categories' : ["z", None]},
                                    ])
    elif model_type == "xgb":
        space = DesignSpace().parse([{'name': 'lr', 'type' : 'num', 'lb' : 0.00005, 'ub' : 0.5},
                                    {'name': 'n_estimators', 'type' : 'int', 'lb' : 1, 'ub' : 20},  # multiplied by 10
                                    {'name': 'max_depth', 'type' : 'int', 'lb' : 2, 'ub' : 10},
                                    {'name': 'subsample', 'type' : 'num', 'lb' : 0.5, 'ub' : 0.99},
                                    {'name': 'colsample_bytree', 'type' : 'num', 'lb' : 0.5, 'ub' : 0.99},
                                    {'name': 'gamma', 'type' : 'num', 'lb' : 0.01, 'ub' : 5.0},
                                    {'name': 'min_child_weight', 'type' : 'num', 'lb' : 0.01, 'ub' : 5},
                                    
                                    {'name': 'fill_type', 'type' : 'cat', 'categories' : ['median', 'none']},
                                    {'name': 'flat_block_size', 'type' : 'int', 'lb' : 1, 'ub' : 4}
                                    ])
        #cfg["flat_block_size"] = 8
    return space


def tune_hebo(df, space, cfg):
    opt = HEBO(space, rand_sample=0,
            model_name="gpy")#"rf")#"gpy")

    opt_steps = cfg["opt_steps"]

    cfg["verbose"] = False

    for i in range(opt_steps):
        rec = opt.suggest()
        print(i)
        print(list(zip(rec.columns, rec.values[0])))
        opt.observe(rec, obj_hebo(df, cfg, rec))
        min_idx = np.argmin(opt.y)
        print("Current score:", 1 - opt.y[-1][0])
        print(f'After {i} iterations, best obj is {1 - opt.y[min_idx][0]:.4f}')
        print()

    opt_df = opt.X
    opt_df["y"] = opt.y
    opt_df["y"].plot()
    opt_df["score"] = 1 - opt_df["y"]

    plt.show()
    opt_df.plot.scatter(x="lr", y="score")
    plt.show()

    # create a folder to save the results
    import os
    import datetime
    # create folder name according to the database name, minutes, model type and date
    folder_name = f"hebo_tunings/{cfg['db_name']}_{cfg['minutes']}/{cfg['model_type']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(folder_name, exist_ok=True)
    # save the results
    opt_df.to_csv(f"{folder_name}/results.csv", index=False)


    # one-hot encode
    input_df = opt.X.drop(columns=["y"])
    if "score" in input_df:
        input_df = input_df.drop(columns=["score"])

    if "fill_type" in input_df:
        one_hot_fill = pd.get_dummies(opt.X.fill_type, prefix='fill')
        input_df = pd.concat([input_df, one_hot_fill], axis=1).drop(columns=["fill_type"]).astype(float)
    else:
        input_df = input_df.astype(float)

    # feat importance using rf
    rf = sklearn.ensemble.RandomForestRegressor(100)
    rf.fit(input_df, 1 - opt.y)
    pd.Series(data=rf.feature_importances_, index=input_df.columns).sort_values()

    import shap
    expl = shap.TreeExplainer(rf, data=input_df, model_output='raw', 
                            feature_perturbation='interventional')
    shap_vals = expl.shap_values(input_df, check_additivity=False)
    shap.summary_plot(shap_vals, input_df.astype(float))
    plt.show()

    mean_shap_vals = np.abs(shap_vals).mean(axis=0)
    mean_shap_vals /= mean_shap_vals.sum()
    pd.Series(data=mean_shap_vals, index=input_df.columns).sort_values().plot.bar()
    plt.show()
