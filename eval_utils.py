import pandas as pd
import numpy as np
import torch
import sklearn


def get_dl(data_module, dl_type):
    if dl_type == "test":
        dl = data_module.test_dataloader()
    elif dl_type == "train":
        dl = data_module.train_dataloader()
        dl.dataset.train = False
    else:
        dl = data_module.val_dataloader()
    return dl


def get_mean_train_target(data_module):
    return np.concatenate([pat[~torch.isnan(pat)].numpy() for pat in data_module.train_ds.targets]).mean()


def make_pred_df(model, dl_type="test", dl=None, calc_new_norm_stats=False):
    # Eval model
    # prep
    all_targets = []
    all_preds = []
    all_times = []
    all_ids = []
    all_losses = []
    count = 0
    model.to("cuda")
    model.eval()
    # get dataloader
    if dl is None:
        dl = get_dl(model.data_module, dl_type)
    else:
        # normalize inputs if the dl was not used in datamodule
        if not hasattr(dl.dataset, "was_normed") or dl.dataset.was_normed == False:
            if calc_new_norm_stats:
                mean, std = model.data_module.get_mean_std(dl)
                dl.dataset.inputs = [model.data_module.fill(pat, mean, fill_type=dl.fill_type) for pat in dl.dataset.raw_inputs]
                dl.dataset.inputs = [(pat - mean) / std for pat in dl.dataset.inputs]
            else:
                dl.dataset.inputs = [model.data_module.preprocess(pat) for pat in dl.dataset.raw_inputs]
            dl.dataset.was_normed = True
    
    # make preds
    for inputs, targets, lens in dl:
        bs = inputs.shape[0]
        # to gpu
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        # pred
        with torch.no_grad():
            preds = model(inputs)
        # loss
        loss = model.loss_func(preds, targets)
        # other details
        times = torch.stack([torch.arange(inputs.shape[1]) for _ in range(bs)]).unsqueeze(-1)
        ids = torch.stack([torch.ones(inputs.shape[1]) * (count + i) for i in range(bs)]).unsqueeze(-1)
        count += bs

        targets = torch.cat([t[:l] for t, l in zip(targets, lens)]).flatten().cpu()
        times = torch.cat([t[:l] for t, l in zip(times, lens)]).flatten().cpu()
        preds = torch.cat([t[:l] for t, l in zip(preds, lens)]).flatten().cpu()
        ids = torch.cat([t[:l] for t, l in zip(ids, lens)]).flatten().cpu()

        all_targets.append(targets)
        all_preds.append(preds)
        all_times.append(times)
        all_ids.append(ids)
        all_losses.append(loss)
    model.to("cpu")
    if dl_type == "train":
        dl.dataset.train = True

    all_losses = torch.stack(all_losses).cpu().flatten().numpy()
    all_targets = torch.cat(all_targets).cpu().flatten().numpy()
    all_preds = torch.cat(all_preds).cpu().flatten().numpy()
    all_ids = torch.cat(all_ids).cpu().flatten().numpy()
    all_times = torch.cat(all_times).cpu().flatten().numpy()
    all_errors = (all_targets - all_preds) ** 2
    

    df = pd.DataFrame({"targets": all_targets, "preds": all_preds, "ids": all_ids, "step": all_times, "error": all_errors})
    df["mean_train_target"] = model.data_module.mean_train_target  #get_mean_train_target(model.data_module)
    return df


def make_pred_df_xgb(model, data_module, regression, dl_type="test", dl=None, calc_new_norm_stats=False):
    # get inputs+targets
    if dl is None:
        dl = get_dl(data_module, dl_type)
    else:
        dl.dataset.preprocess_all(data_module.fill_type, 
                                  median=data_module.medians, 
                                  mean=data_module.means, 
                                  std=data_module.stds,
                                  upper_quants=data_module.upper_quants,
                                  lower_quants=data_module.lower_quants)
    inputs = dl.dataset.flat_inputs
    targets = dl.dataset.flat_targets
    # get patient ids and steps
    ids = dl.dataset.ids
    steps = dl.dataset.steps
    # predict
    if regression:
        preds = model.predict(inputs)
    else:
        preds = model.predict_proba(inputs)[:, 1]
    errors = (preds - targets) ** 2
    df = pd.DataFrame({"targets": targets, "preds": preds, "ids": ids, "step": steps, "error": errors})
    df["mean_train_target"] = data_module.mean_train_target
    return df

def mape(targets, preds):
    target_sum = np.sum(targets)
    error_sum = np.sum(np.abs(targets - preds))
    if target_sum == 0 and error_sum == 0:
        return 0
    if target_sum == 0:
        return 1
    else:
        return error_sum / target_sum
    #epsilon = np.finfo(np.float64).eps
    #return np.mean(np.abs((preds - targets)) / np.max(np.abs(targets) + epsilon))
    
def hypertension_acc(targets, preds):
    thresh = 22
    hyper_targets = targets > thresh
    hyper_preds = preds > thresh
    hyper_acc = (hyper_targets == hyper_preds).astype(float).mean()
    return hyper_acc


def hypertension_prec_rec(targets, preds):
    thresh = 22
    hyper_targets = targets > thresh
    hyper_preds = preds > thresh
    CP = hyper_targets == 1
    CN = hyper_targets == 0
    TP = (hyper_targets[CP] == hyper_preds[CP]).sum()
    TN = (hyper_targets[CN] == hyper_preds[CN]).sum()
    FP = (hyper_targets[CN] != hyper_preds[CN]).sum()
    FN = (hyper_targets[CP] != hyper_preds[CP]).sum()
    sens = TP / (TP + FN) if (TP + FN) != 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) != 0 else np.nan
    prec = TP / (TP + FP) if (TP + FP) != 0 else np.nan
    rec = sens
    return prec, rec


def calc_metric(model_df):
    # calcs RMSE per patient and averages them
    return model_df.groupby("ids").apply(lambda pat: 
                                         np.sqrt(
                                             ((pat["targets"] - pat["preds"]) ** 2).mean()
                                         )
                                        ).mean()

def r2_score(targets, preds, baseline_target=None):
    if baseline_target is None:
        baseline_target = np.mean(targets)
    return 1 - (np.sum((targets - preds) ** 2) / np.sum((targets - baseline_target) ** 2))


def print_all_metrics(df):
    mean_train_target = df["mean_train_target"].iloc[0]
        
    print("Performance over splits: ")
    mean_df = df.groupby("model_id").mean()[["targets", "preds", "error"]]
    print(mean_df)
    df_nona = df[~df["targets"].isna()]
    targets = df_nona["targets"]
    preds = df_nona["preds"]
    mean_pred = preds.mean()
    std_pred = preds.std()
    mean_target = targets.mean()
    std_target = targets.std()
    print("Mean train target: ", mean_train_target)
    print("Mean/Std preds: ", mean_pred, std_pred)
    print("Mean/Std targets: ", mean_target, std_target)
    print("Max error: ", np.max(df_nona["error"]))
    print("Accuracy for hypertension baseline: ", hypertension_acc(targets, np.zeros((len(targets,)))))
    print()
        
    test_target_mse = sklearn.metrics.mean_squared_error(targets, np.ones(len(targets)) * mean_target)
    pred_mse = mean_df["error"].mean()

    print("Model metrics:")
    print("RMSE: ", np.sqrt(pred_mse))
    print("MSE: ", pred_mse)
    print("MAE: ", df_nona.groupby("model_id").apply(lambda pat: sklearn.metrics.mean_absolute_error(pat["targets"], pat["preds"])).mean())
    print("MAPE: ", df_nona.groupby("model_id").apply(lambda pat: mape(pat["targets"], pat["preds"])).mean())
    #print("all R2", df_nona.groupby("ids").apply(lambda pat: r2_score(pat["targets"], pat["preds"], baseline_target=mean_target)))
    print("R2 custom: ", 1 - pred_mse / test_target_mse)
    print("R2: ", df_nona.groupby("model_id").apply(lambda pat: r2_score(pat["targets"], pat["preds"], baseline_target=mean_target)).mean())
    print("R2 old: ", sklearn.metrics.r2_score(targets, preds))
    print("Accuracy for hypertension: ", hypertension_acc(targets, preds).mean())#.groupby("model_id").apply(lambda pat: hypertension_acc(pat["targets"], pat["preds"])).mean())
    print("Precision for hypertension: ", df_nona.groupby("model_id").apply(lambda pat: hypertension_prec_rec(pat["targets"], pat["preds"])[0]).mean())
    print("Recall for hypertension: ", df_nona.groupby("model_id").apply(lambda pat: hypertension_prec_rec(pat["targets"], pat["preds"])[1]).mean())
    print()

    train_target_mse = df.groupby("model_id").apply(lambda pat: (pat["targets"] - mean_train_target).pow(2).mean()).mean()
    print("Mean train baseline metrics:")
    print("Mean train target:", mean_train_target)
    print("RMSE: ", np.sqrt(train_target_mse))
    print("MSE: ", train_target_mse)
    #print("Loss: ", sklearn.metrics.mean_squared_error(targets, baseline_preds))
    print("MAE: ", df_nona.groupby("model_id").apply(lambda pat: sklearn.metrics.mean_absolute_error(pat["targets"], np.ones(len(pat)) * mean_train_target)).mean())
    print("MAPE: ", df_nona.groupby("model_id").apply(lambda pat: mape(pat["targets"], np.ones(len(pat)) * mean_train_target)).mean())
    print("R2 custom: ", 1 - train_target_mse / mean_target)
    print("R2 score: ", df_nona.groupby("model_id").apply(lambda pat: r2_score(pat["targets"], np.ones(len(pat)) * mean_train_target, baseline_target=mean_target)).mean())
    print()
    print()
    print("Scores for test target mean")
    print("Mean test target: ", mean_target)
    print("MSE: ", test_target_mse)
    print("MAE: ", df.groupby("model_id").apply(lambda pat: np.abs(pat["targets"] - mean_target).mean()).mean())
    print("R2 score: ", 0)

def print_all_metrics_pat(df):
    mean_train_target = df["mean_train_target"].iloc[0]
        
    print("Performance over splits: ")
    mean_df = df.groupby("model_id").apply(lambda model_df: model_df.groupby("ids").mean().mean())[["targets", "preds", "error"]]
    print(mean_df)
    df_nona = df[~df["targets"].isna()]
    targets = df_nona["targets"]
    preds = df_nona["preds"]
    mean_pred = mean_df["preds"].mean()
    std_pred = mean_df["preds"].std()
    mean_target = mean_df["targets"].mean()
    std_target = mean_df["targets"].std()
    print("Mean train target: ", mean_train_target)
    print("Mean/Std preds: ", mean_pred, std_pred)
    print("Mean/Std targets: ", mean_target, std_target)
    print("Max error: ", np.max(df.groupby("ids").mean()["error"]))
    print("Accuracy for hypertension baseline: ", hypertension_acc(df_nona["targets"], np.ones(len(pat))))
    print()
    
    test_target_mse = df.groupby("ids").apply(lambda pat: (pat["targets"] - mean_target).pow(2).mean()).mean()
    pred_mse = df.groupby("ids").apply(lambda pat: pat["error"].mean()).mean()


    print("Model metrics:")
    print("RMSE: ", np.sqrt(pred_mse))
    print("MSE: ", pred_mse)
    print("MAE: ", df_nona.groupby("ids").apply(lambda pat: sklearn.metrics.mean_absolute_error(pat["targets"], pat["preds"])).mean())
    print("MAPE: ", df_nona.groupby("ids").apply(lambda pat: mape(pat["targets"], pat["preds"])).mean())
    #print("all R2", df_nona.groupby("ids").apply(lambda pat: r2_score(pat["targets"], pat["preds"], baseline_target=mean_target)))
    print("R2 custom: ", 1 - pred_mse / test_target_mse)
    print("R2: ", df_nona.groupby("ids").apply(lambda pat: r2_score(pat["targets"], pat["preds"], baseline_target=mean_target)).mean())
    print("Accuracy for hypertension: ", df_nona.groupby("ids").apply(lambda pat: hypertension_acc(pat["targets"], pat["preds"])).mean())
    print("Precision for hypertension: ", df_nona.groupby("ids").apply(lambda pat: hypertension_prec_rec(pat["targets"], pat["preds"])[0]).mean())
    print("Recall for hypertension: ", df_nona.groupby("ids").apply(lambda pat: hypertension_prec_rec(pat["targets"], pat["preds"])[1]).mean())
    print()

    train_target_mse = df.groupby("ids").apply(lambda pat: (pat["targets"] - mean_train_target).pow(2).mean()).mean()
    print("Mean train baseline metrics:")
    print("Mean train target:", mean_train_target)
    print("RMSE: ", np.sqrt(train_target_mse))
    print("MSE: ", train_target_mse)
    #print("Loss: ", sklearn.metrics.mean_squared_error(targets, baseline_preds))
    print("MAE: ", df_nona.groupby("ids").apply(lambda pat: sklearn.metrics.mean_absolute_error(pat["targets"], np.ones(len(pat)) * mean_train_target)).mean())
    print("MAPE: ", df_nona.groupby("ids").apply(lambda pat: mape(pat["targets"], np.ones(len(pat)) * mean_train_target)).mean())
    print("R2 custom: ", 1 - train_target_mse / mean_target)
    print("R2 score: ", df_nona.groupby("ids").apply(lambda pat: r2_score(pat["targets"], np.ones(len(pat)) * mean_train_target, baseline_target=mean_target)).mean())
    print()
    print()
    print("Scores for test target mean")
    print("Mean test target: ", mean_target)
    print("MSE: ", test_target_mse)
    print("MAE: ", df.groupby("ids").apply(lambda pat: np.abs(pat["targets"] - mean_target).mean()).mean())
    print("R2 score: ", 0)



def get_all_dfs(models, trainers, model_type, regression, dl_type="test", dl=None, calc_new_norm_stats=False):
    classical_models = ["linear", "xgb", "rf"]
    if model_type in classical_models:
        dfs = [make_pred_df_xgb(model, data_module, regression, dl_type=dl_type, dl=dl, calc_new_norm_stats=calc_new_norm_stats) for model, data_module in zip(models, trainers)]
    else:
        dfs = [make_pred_df(model, dl_type=dl_type, dl=dl, calc_new_norm_stats=calc_new_norm_stats) for model in models]
    for i in range(len(dfs)):
        dfs[i]["model_id"] = i
    return pd.concat(dfs)
