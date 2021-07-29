import torch
from src.utils import pop_feature


class SequentialLoss:
    def __init__(self, categorical_feature_idcs, class_weights, target_names,
                 norm_seq_len):
        self.categorical_feature_idcs = categorical_feature_idcs
        self.target_names = target_names
        self.norm_seq_len = norm_seq_len
        self.class_weights = class_weights
        self.cat_loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.regression_loss_func = torch.nn.MSELoss(reduction='mean')

    def __call__(self, pred, target, mask):
        """Calculates the loss and considers missing values in the loss given by mask"""
        # shape: [BS, LEN, FEATS]
        # Apply mask:
        num_feats = target.shape[-1]
        mask = ~mask
        pred_per_pat = [pred_p[mask_p].reshape(-1, num_feats) for pred_p, mask_p in zip(pred, mask) if sum(mask_p) > 0]
        target_per_pat = [target_p[mask_p].reshape(-1, num_feats) for target_p, mask_p in zip(target, mask) if sum(mask_p) > 0]
        
        # calc raw loss per patient
        if self.categorical_feature_idcs:
            if num_feats > len(self.categorical_feature_idcs):
                # here I could define a loss function to mix categorical and non categorical 
                pass 
            else:
                # for now simply apply mean categorical loss per patient
                loss_per_pat = [self.cat_loss_func(p, t) for p, t in zip(pred_per_pat, target_per_pat)]
        else:
            loss_per_pat = [self.regression_loss_func(p, t) for p, t in zip(pred_per_pat, target_per_pat)]
        
        loss = torch.stack(loss_per_pat).mean()
        
        verbose = False
        if verbose:
            print("pred per pat", pred_per_pat)
            print(" target per pat", target_per_pat)
            print("loss per pat", loss_per_pat)
            print("loss: ", loss)
            print()
        return loss
        
        

        #pred[mask] = 0.0
        #target[mask] = 0.0
        #step_mask = mask.sum(-1).bool()
        


        # Pop out the categoricals:
        if self.categorical_feature_idcs:
            categorical_loss = 0
            pred_categorical, pred = pop_feature(pred, self.categorical_feature_idcs)
            target_categorical, target = pop_feature(target, self.categorical_feature_idcs)
            for idx, cat_idx in enumerate(self.categorical_feature_idcs):
                cat_preds = pred_categorical[:, :, idx]
                cat_targets = target_categorical[:, :, idx]
                cat_loss = self.cat_loss_fn(cat_preds, cat_targets)
                if self.class_weights:
                    cat_loss[cat_targets == 1] *= self.class_weights[cat_idx]
                categorical_loss += cat_loss
            #categorical_loss[step_mask] = 0

        # Calculate the loss of the regression on all other features:
        rest_loss = self.regression_loss_fcn(pred, target)
        #rest_loss[step_mask] = 0

        sum_rest_loss = rest_loss.sum(dim=-1).sum(dim=-1)
        sum_categorical_loss = 0.0
        if self.categorical_feature_idcs:
            sum_categorical_loss = categorical_loss.sum(-1)#.sum(-1)   

        # Norm loss per seq len and per non NAN targets.
        # Basically, this reduces the weight of longer sequences and of sequences with more NaNs.
        # Add 1 to avoid zero division:
        count_per_pat = 1
        if self.norm_seq_len:
            count_per_pat = 1#(~step_mask).sum(dim=-1) + 1
            

        # Aggregate losses:
        loss = ((sum_rest_loss + sum_categorical_loss) /
                count_per_pat).mean()
        
        verbose = False
        if verbose:
            print(pred.shape)
            print(target.shape)
            print(mask.shape, mask.sum(), mask.float().mean())
            print(step_mask.shape, step_mask.sum(), step_mask.float().mean())
            print("rest loss shape: ", rest_loss.shape)
            print("sum rest loss. ", sum_rest_loss)
            print("cat loss. ", sum_categorical_loss)
            print("count per pat: ", count_per_pat)
            print(loss)
        return loss
