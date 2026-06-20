import json
import numpy as np
import pandas as pd
from pomegranate import BayesianNetwork
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, silhouette_score, matthews_corrcoef
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import torch
from scipy.stats import wasserstein_distance_nd
import torch.nn as nn
import torch.nn.functional as F
from octgan.constants import CATEGORICAL, CONTINUOUS, ORDINAL

_MODELS = {
    'binary_classification': [
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
                'max_depth': 20
            }
        },
        {
            'class': AdaBoostClassifier,
        },
        {
            'class': LogisticRegression,
            'kwargs': {
                'solver': 'lbfgs',
                'n_jobs': -1,
                'max_iter': 50
            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (50, ),
                'max_iter': 50
            },
        }
    ],
    'multiclass_classification': [
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
                'max_depth': 30,
                'class_weight': 'balanced',
            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (100, ),
                'max_iter': 50
            },
        }
    ],
    'regression': [
        {
            'class': LinearRegression,
        },
        {
            'class': MLPRegressor,
            'kwargs': {
                'hidden_layer_sizes': (100, ),
                'max_iter': 50
            },
        }
    ],

    'clustering': [
        {
            'class': KMeans, 
            'kwargs': {
                'n_clusters': 2,
                'n_jobs': -1,
            }
        }
    ]
}


class FeatureMaker:

    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):

        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]

        features = []
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    labels = col.astype(int)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col, 1e-2))

                else:
                    feature = (col - cmin) / (cmax - cmin) 

            elif cinfo['type'] == ORDINAL:
                feature = col

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features, labels


def _prepare_ml_problem(train, test, metadata, clustering=False): 
    fm = FeatureMaker(metadata)
    x_train, y_train = fm.make_features(train)
    x_test, y_test = fm.make_features(test)
    if clustering:
        model = _MODELS["clustering"]
    else:
        model = _MODELS[metadata['problem_type']]
    return x_train, y_train, x_test, y_test, model


def _evaluate_multi_classification(train, test, metadata):
   
    """Score classifiers using f1 score and the given train and test data.

    Args:
        x_train(numpy.ndarray):
        y_train(numpy.ndarray):
        x_test(numpy.ndarray):
        y_test(numpy):
        classifiers(list):

    Returns:
        pandas.DataFrame
    """
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)

    performance = []
    f1 = [] 
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

        report = classification_report(y_test, pred, output_dict=True)
        classes = list(report.keys())[:-3]
        proportion = [  report[i]['support'] / len(y_test) for i in classes]
        weighted_f1 = np.sum(list(map(lambda i, prop: report[i]['f1-score']* (1-prop)/(len(classes)-1), classes, proportion)))
                
        f1.append([report[c]['f1-score'] for c in classes] )
        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average='macro')
        micro_f1 = f1_score(y_test, pred, average='micro')

        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                'weighted_f1': weighted_f1,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1
            }
        )

    return pd.DataFrame(performance)


def _evaluate_binary_classification(train, test, metadata):
   
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)
    performance = []
    f1 = [] 
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
            pred_prob = np.array([1.] * len(x_test))

        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred_prob = model.predict_proba(x_test)


        acc = accuracy_score(y_test, pred)
        binary_f1 = f1_score(y_test, pred, average='binary')
        macro_f1 = f1_score(y_test, pred, average='macro')
        report = classification_report(y_test, pred, output_dict=True)
        classes = list(report.keys())[:-3]

        f1.append([report[c]['f1-score'] for c in classes] )

        mcc = matthews_corrcoef(y_test, pred)

        precision = precision_score(y_test, pred, average='binary')
        recall = recall_score(y_test, pred, average='binary')
        size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
        rest_label = set(range(size)) - set(unique_labels)
        tmp = []
        j = 0
        for i in range(size):
            if i in rest_label:
                tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
            else:
                try:
                    tmp.append(pred_prob[:,[j]])
                except:
                    tmp.append(pred_prob[:, np.newaxis])
                j += 1
        roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))

        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                "binary_f1": binary_f1,
                "macro_f1": macro_f1,
                "matthews_corrcoef": mcc, 
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc
            }
        )
    
    return pd.DataFrame(performance)

def _evaluate_regression(train, test, metadata):
   
    x_train, y_train, x_test, y_test, regressors = _prepare_ml_problem(train, test, metadata)

    performance = []
    y_train = np.log(np.clip(y_train, 1, 20000))
    y_test = np.log(np.clip(y_test, 1, 20000))
    for model_spec in regressors:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        r2 = r2_score(y_test, pred)
        explained_variance = explained_variance_score(y_test, pred)
        mean_squared = mean_squared_error(y_test, pred)
        mean_absolute = mean_absolute_error(y_test, pred)



        performance.append(
            {
                "name": model_repr,
                "r2": r2,
                "explained_variance" : explained_variance,
                "mean_squared_error" : mean_squared,
                "mean_absolute_error" : mean_absolute
            }
        )

    return pd.DataFrame(performance)

def _evaluate_gmm_likelihood(train, test, metadata, components=[10, 30]):
    results = list()
    for n_components in components:
        gmm = GaussianMixture(n_components, covariance_type='diag')
        gmm.fit(test)
        l1 = gmm.score(train)

        gmm.fit(train)
        l2 = gmm.score(test)

        results.append({
            "name": repr(gmm),
            "syn_likelihood": l1,
            "test_likelihood": l2,
        })

    return pd.DataFrame(results)

def _mapper(data, metadata):
    data_t = []
    for row in data:
        row_t = []
        for id_, info in enumerate(metadata['columns']):
            row_t.append(info['i2s'][int(row[id_])])

        data_t.append(row_t)

    return data_t

def _evaluate_bayesian_likelihood(train, test, metadata):
    structure_json = json.dumps(metadata['structure'])
    bn1 = BayesianNetwork.from_json(structure_json)

    train_mapped = _mapper(train, metadata)
    test_mapped = _mapper(test, metadata)
    prob = []
    for item in train_mapped:
        try:
            prob.append(bn1.probability(item))
        except Exception:
            prob.append(1e-8)

    l1 = np.mean(np.log(np.asarray(prob) + 1e-8))

    bn2 = BayesianNetwork.from_structure(train_mapped, bn1.structure)
    prob = []

    for item in test_mapped:
        try:
            prob.append(bn2.probability(item))
        except Exception:
            prob.append(1e-8)

    l2 = np.mean(np.log(np.asarray(prob) + 1e-8))

    return pd.DataFrame([{
        "name": "Bayesian Likelihood",
        "syn_likelihood": l1,
        "test_likelihood": l2,
    }])


def _evaluate_cluster(train, test, metadata):
   
    x_train, y_train, x_test, y_test, kmeans = _prepare_ml_problem(train, test, metadata, clustering=True)
 

    model_class = kmeans[0]['class']
    model_repr = model_class.__name__
    unique_labels = np.unique(y_train)
    num_columns = metadata['columns'][-1]["size"]
    
    result = []
    for i in range(3):
        model = model_class(n_clusters = num_columns*(i+1))

        if len(unique_labels) == 1:
            result.append([unique_labels[0]] * len(x_test))

        else:
            try:
                model.fit(x_train)
                predicted_label = model.predict(x_test)
            except:
                x_train = x_train.astype(np.float32)
                model.fit(x_train)

                x_test = x_test.astype(np.float32)
                predicted_label = model.predict(x_test)
            try:
                result.append(silhouette_score(x_test, predicted_label, metric='euclidean', sample_size=100))
            except:
                result.append(0)
        

    return pd.DataFrame([{
        "name": model_repr,
        "silhouette_score": np.mean(result),
    }])



_EVALUATORS = {
    'bayesian_likelihood': [_evaluate_bayesian_likelihood],
    'gaussian_likelihood': [_evaluate_gmm_likelihood],
    'regression': [_evaluate_regression],
    'binary_classification': [_evaluate_binary_classification, _evaluate_cluster],
    'multiclass_classification': [_evaluate_multi_classification, _evaluate_cluster]
}

def compute_scores(test, synthesized_data, metadata):
    result = pd.DataFrame()

    for evaluator in _EVALUATORS[metadata['problem_type']]:
        scores = pd.DataFrame()
        
        for i in range(5):
            score = evaluator(synthesized_data, test, metadata) 
            score['test_iter'] = i
            scores = pd.concat([scores, score], ignore_index=True)
        scores = scores.groupby(['test_iter']).mean() 
        result = pd.concat([result, scores], axis=1)

    return result


def trajectory_metrics(generated_trajectories, real_trajectories):
    ade_values = []
    for gen_traj, real_traj in zip(generated_trajectories, real_trajectories):
        displacement_errors = torch.norm(gen_traj - real_traj, p=2, dim=1)
        ade_values.append(torch.mean(displacement_errors).item())
    return float(np.mean(ade_values)) if ade_values else 0.0


def cosine_similarity_metric(all_samples, test_x):
        if all_samples.shape != test_x.shape:
            return 0.0

        if all_samples.shape[1] % 3 != 0:
            return 0.0

        samples = all_samples.detach().cpu()
        real = test_x.detach().cpu()
        region_dim = samples.shape[1] // 3

        sample_blocks = [samples[:, :region_dim], samples[:, region_dim:2 * region_dim], samples[:, 2 * region_dim:]]
        real_blocks = [real[:, :region_dim], real[:, region_dim:2 * region_dim], real[:, 2 * region_dim:]]

        cos_values = []
        for idx, (sample_block, real_block) in enumerate(zip(sample_blocks, real_blocks)):
            min_vals = real_block.min(dim=0).values
            max_vals = real_block.max(dim=0).values
            denom = (max_vals - min_vals)
            denom = torch.where(denom > 1e-8, denom, torch.ones_like(denom))

            sample_norm = (sample_block - min_vals) / denom
            real_norm = (real_block - min_vals) / denom

            if idx == 2:
                sample_norm = 1.0 - sample_norm
                real_norm = 1.0 - real_norm

            cos = F.cosine_similarity(sample_norm, real_norm, dim=1)
            if cos.numel() > 0:
                cos_values.append(cos.mean().item())

        if not cos_values:
            return 0.0
        return float(np.mean(cos_values))


def metrics(all_samples, test_x, trajectory_lengths):
    '''
    all_samples: sampled data (flat 2D tensor)
    test_x: test data (flat DataFrame or numpy array)
    trajectory_lengths: A list/array indicating the number of timepoints for each subject.
    '''

    def kld(p, q):
        p = p + 1e-10
        return (p * (p.log() - q.log())).sum(dim=1)

    # Convert test_x DataFrame to a tensor, handling categorical/target columns
    if isinstance(test_x, pd.DataFrame):
        # Assuming numerical columns are identifiable (e.g., by name)
        num_cols = [c for c in test_x.columns if 'num' in c]  # Adjust this logic as needed
        real_data_tensor = torch.tensor(test_x[num_cols].values, dtype=torch.float32)
    else:  # Assume it's already a numpy array or similar
        real_data_tensor = torch.tensor(test_x, dtype=torch.float32)

    # --- Reconstruct Trajectories for ADE ---
    # `torch.split` uses the trajectory_lengths to split the flat tensor back into a list of tensors
    generated_trajectories = list(torch.split(all_samples, trajectory_lengths))
    real_trajectories = list(torch.split(real_data_tensor, trajectory_lengths))

    ade = trajectory_metrics(generated_trajectories, real_trajectories)

    # --- Calculate Metrics on Flat Data ---
    if (all_samples.min() < -1e+10) or (all_samples.max() > 1e+10):
        print('sampled values are exploded !!!')
        nrmse, rmse, jsd_mean, wd, cos_sim = [None] * 5
    else:
        sampled_data_np = all_samples.cpu().numpy()
        real_data_np = real_data_tensor.cpu().numpy()

        wd = max(0.0, float(wasserstein_distance_nd(sampled_data_np, real_data_np)))

        sampled_data_softmax = F.softmax(all_samples, dim=1)
        real_data_softmax = F.softmax(real_data_tensor, dim=1)
        m = 0.5 * (sampled_data_softmax + real_data_softmax)
        jsd = 0.5 * (kld(sampled_data_softmax, m) + kld(real_data_softmax, m))
        jsd_mean = jsd.mean().item()

        mse_val = F.mse_loss(all_samples, real_data_tensor)
        rmse = torch.sqrt(mse_val).item()

        gt_range = real_data_tensor.max() - real_data_tensor.min()
        nrmse = rmse / gt_range.item() if gt_range > 1e-8 else float('inf')

        cos_sim = cosine_similarity_metric(all_samples, real_data_tensor)

    return nrmse, rmse, jsd_mean, wd, ade, cos_sim


def sample_metrics(X_train, X_test, synthesizer, split='val'):
    X_test_lengths = [2, 2, 2, 2, 3, 2, 3, 3, 3, 4, 3, 2, 3, 2, 2, 3, 2, 3]
    batch_size = 1000
    device = 'cuda'
    if split == 'val':
        y = np.array(X_test['target'])
        X_num_shape = len([c for c in X_test.columns.tolist() if 'num' in c])
        X_cat = np.stack([np.array(X_test[c]) for c in X_test.columns.tolist() if 'cat' in c], 1)
    else:
        y = np.array(X_train['target'])
        X_num_shape = len([c for c in X_train.columns.tolist() if 'num' in c])
        X_cat = np.stack([np.array(X_train[c]) for c in X_train.columns.tolist() if 'cat' in c], 1)

    sample_gen = torch.zeros(y.shape[0], X_num_shape).cpu()
    remain_sample_id = torch.arange(len(sample_gen))
    org_X_cat = torch.from_numpy(X_cat.astype(int)).to(device)
    org_y = torch.from_numpy(y).to(device)
    max_iter = 10
    sample_iter = 0
    while len(remain_sample_id) > 0:
        if sample_iter > max_iter: break
        sample_iter += 1
        print(f'Remain {len(remain_sample_id)} Samples being generating')
        X_cat = org_X_cat[remain_sample_id.to(device)]
        y = org_y[remain_sample_id.to(device)]
        # Create synthetic data
        gen_data = synthesizer.sample(batch_size)

        gen_y = np.array(gen_data['target'])
        gen_X_cat = np.stack([np.array(gen_data[c]) for c in gen_data.columns.tolist() if 'cat' in c], 1)

        X_num = np.stack([np.array(gen_data[c]) for c in gen_data.columns.tolist() if 'num' in c], 1)

        X_num = torch.from_numpy(X_num.astype(float))
        gen_X_cat = torch.from_numpy(gen_X_cat.astype(int)).to(device)
        gen_y = torch.from_numpy(gen_y.astype(int)).to(device)
        batch_mask_cat_cond = torch.any(X_cat[:, None] != gen_X_cat[None, :], dim=-1).cpu()  # N_y x N_sample
        batch_mask_y_cond = (y[:, None] != gen_y[None, :]).cpu()  # N_y x N_sample
        batch_mask_cond = torch.logical_or(batch_mask_y_cond, batch_mask_cat_cond)
        mask_cond = batch_mask_cond.all(dim=1)  # N_y
        all_yi = torch.where(~mask_cond)[0]
        yi_ls, samplei_ls = torch.where(~batch_mask_cond)
        sample_ind = []
        for yi, samplei in zip(yi_ls, samplei_ls):
            if yi in all_yi:
                sample_ind.append(samplei)
                all_yi = all_yi[all_yi != yi]

        sample_ind = torch.LongTensor(sample_ind)
        sample_gen[remain_sample_id[~mask_cond]] = X_num[sample_ind].float()
        remain_sample_id = remain_sample_id[mask_cond]

    if split == 'val':
        # Pass the trajectory_lengths to the metrics function
        nrmse, rmse, jsd_mean, wd, ade, cos_sim = metrics(sample_gen, X_test, X_test_lengths)
    else:
        # Pass the (hypothetical) train lengths
        # wd, jsd_mean, rmse, mae, nrmse, nmse, ade, fde = metrics(sample_gen, X_train, trajectory_lengths_train)
        pass

    return pd.DataFrame([{
        "name": "gen metrics",
        "nrmse": nrmse,
        "rmse": rmse,
        "jsd": jsd_mean,
        "wd": wd,
        "ade": ade,
        "cossim": cos_sim,
    }])
