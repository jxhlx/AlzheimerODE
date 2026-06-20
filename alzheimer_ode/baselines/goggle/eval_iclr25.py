import torch, os
import argparse
from pathlib import Path
from scipy.stats import wasserstein_distance_nd
import torch.nn.functional as F
import numpy as np
# import lib

ROOT_DIR = Path(__file__).resolve().parent


class Evaluator:
    def metrics(self, all_samples, test_x):
        '''
        all_samples: sampled data
        test_x: test data (i.e., brain measurments)

        The input shapes of all_samples and test_x are 2D and both are same: batch x (num_visits x num_node)   
        '''
        # print(all_samples.shape, test_x.shape)
        # exit()
        if (all_samples.min() < -1e+10) or (all_samples.max() > 1e+10):
            print('sampled values are exploded !!!')
            return None
        else:

            test_x = test_x[all_samples.mean(axis=-1) > 0]
            all_samples = all_samples[all_samples.mean(axis=-1) > 0]

            # (1) Wasserstein Distance 
            sampled_data_np = all_samples.cpu().numpy()
            real_data_np = test_x.cpu().numpy()

            wd = max(0.0, float(wasserstein_distance_nd(sampled_data_np, real_data_np)))
            # print(f"\nWD: {wd:.3f}")

            # (2) Jensen-Shannon divergence
            sampled_data = F.softmax(all_samples, dim=1)
            real_data = F.softmax(test_x, dim=1)

            m = 0.5 * (sampled_data + real_data) 

            assert not torch.isnan(self.kld(sampled_data, m)).any().item(), "self.kld(sampled_data, m) has nan"

            jsd = 0.5 * (self.kld(sampled_data, m) + self.kld(real_data, m))
            jsd_mean = jsd.mean().item()
            
            # print(f"JSD: {jsd_mean:.3f}")

            # (3) RMSE
            mse = F.mse_loss(all_samples, test_x)
            rmse = torch.sqrt(mse).item()

            gt_range = test_x.max() - test_x.min()
            nrmse = rmse / gt_range.item() if gt_range > 1e-8 else float("inf")
            cos_sim = self.cosine_similarity_metric(all_samples, test_x)

        return {
            "NRMSE": nrmse,
            "RMSE": rmse,
            "JSD": jsd_mean,
            "WD": wd,
            "ADE": 0.0,
            "CosSim": cos_sim,
        }
    
    def cosine_similarity_metric(self, all_samples, test_x):
        if all_samples.shape != test_x.shape:
            return 0.0

        samples = all_samples.detach().cpu()
        real = test_x.detach().cpu()

        real_block = real
        sample_block = samples

        cos_values = []
        min_vals = real_block.min(dim=0).values
        max_vals = real_block.max(dim=0).values
        denom = (max_vals - min_vals)
        denom = torch.where(denom > 1e-8, denom, torch.ones_like(denom))

        sample_norm = (sample_block - min_vals) / denom
        real_norm = (real_block - min_vals) / denom

        cos = F.cosine_similarity(sample_norm, real_norm, dim=1)
        if cos.numel() > 0:
            cos_values.append(cos.mean().item())

        if not cos_values:
            return 0.0
        return float(np.mean(cos_values))


    def kld(self, p, q):
        p = p + 1e-10 
        return (p * (p.log() - q.log())).sum(dim=1)


    def kld(self, p, q):
        p = p + 1e-10 
        return (p * (p.log() - q.log())).sum(dim=1)


def evaluate(real_data_path, gen_data_path, SPLIT = 'train', dn='Amyloid'):
    x_gen = np.load(os.path.join(gen_data_path, f'X_num_{SPLIT}.npy'), allow_pickle=True)#[:, :-1]
    x_real = np.load(os.path.join(real_data_path, f'X_num_{SPLIT}.npy'), allow_pickle=True)#[:, :-1]
    x_gen, x_real = torch.from_numpy(x_gen), torch.from_numpy(x_real)
    eval = Evaluator()
    return eval.metrics(x_gen, x_real)

# methodn = 'goggle'
# print(methodn)
# for split in ['val', 'train']:
#     for dn in ['Amyloid', 'CT', 'FDG', 'Tau']:
#         wds, jsd_means, rmses = [], [], []
#         for i in range(1, 2):
#             real_data_path = f'../data/{dn}/'
#             parent_dir = f'src/tmp/{dn}{i}'
#             wd, jsd_mean, rmse = evaluate(real_data_path, parent_dir, split, dn)
#             wds.append(wd)
#             jsd_means.append(jsd_mean)
#             rmses.append(rmse)
#
#         print(dn, split)
#         print(f'WD: {np.mean(wds):.5f} +- {np.std(wds):.5f}')
#         print(f'JSD: {np.mean(jsd_means):.5f} +- {np.std(jsd_means):.5f}')
#         print(f'RMSE: {np.mean(rmses):.5f} +- {np.std(rmses):.5f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data-path", type=str, default=str(ROOT_DIR / "data" / "ATN"))
    parser.add_argument("--gen-data-template", type=str, default=str(ROOT_DIR / "outputs" / "ATN{}"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    metric_runs = []
    for run_id in range(1, args.runs + 1):
        metrics = evaluate(args.real_data_path, args.gen_data_template.format(run_id), args.split, "ATN")
        if metrics is not None:
            metric_runs.append(metrics)

    print("goggle ATN", args.split)
    for metric_name in ["NRMSE", "RMSE", "JSD", "WD", "ADE", "CosSim"]:
        values = [item[metric_name] for item in metric_runs]
        print(f'{metric_name}: {np.mean(values):.5f} +- {np.std(values):.5f}')


if __name__ == "__main__":
    main()
