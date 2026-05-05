from abc import ABC, abstractmethod
from piq import psnr, ssim, LPIPS
import prettytable
import torch
import torch.nn as nn
import wandb
import numpy as np
import warnings
from analysis_utils import  load_local_tensors, load_true_tensors_by_id, load_local_tensors_by_id ,normalize_tensor, load_tensors_by_id, load_true_images
from torchmetrics.aggregation import MeanMetric, SumMetric

class Evaluator:
    """
        Evaluation module for computing evaluation metrics.
    """

    def __init__(self, eval_fn_list):
        """
            Initializes the evaluator with the ground truth and measurement.

            Parameters:
                eval_fn_list (tuple): List of evaluation functions to use.
        """
        super().__init__()
        self.eval_fn = {}
        for eval_fn in eval_fn_list:
            self.eval_fn[eval_fn.name] = eval_fn
        self.main_eval_fn_name = eval_fn_list[0].name

    def get_main_eval_fn(self):
        """
            return the first eval_fn by default
        """
        return self.eval_fn[self.main_eval_fn_name]

    def __call__(self, gt, measurement, x, reduction='mean'):
        """
            Computes evaluation metrics for the given input.

            Parameters:
                x (torch.Tensor): Input tensor.
                reduction (str): Reduction method ('mean' or 'none').

            Returns:
                dict: Dictionary of evaluation results.
        """
        results = {}
        for eval_fn_name, eval_fn in self.eval_fn.items():
            results[eval_fn_name] = eval_fn(gt, measurement, x, reduction)
        return results

    def to_list(self, x):
        return x.cpu().detach().tolist()
    
    @torch.no_grad()
    def report(self, gt, measurement, x):
        '''x: [N, B, C, H, W] or [B, C, H, W]'''
        if len(x.shape) == 4:
            x = x[None]
        result_dicts = {}

        # eval function
        broadcasted_shape = torch.broadcast_shapes(x.shape, gt.shape)
        x0_flatten = gt.expand(broadcasted_shape).flatten(0, 1)
        x_flatten = x.expand(broadcasted_shape).flatten(0, 1)
        y_flatten = measurement.expand((broadcasted_shape[0], *measurement.shape)).flatten(0, 1)
        with torch.no_grad():
            for key, fn in self.eval_fn.items():
                value = fn(x0_flatten, y_flatten, x_flatten, reduction='none').reshape(broadcasted_shape[0], -1)
                result_dicts[key] = {
                    'sample': self.to_list(value.permute(1, 0)),
                    'mean': self.to_list(value.mean(0)),
                    'std': self.to_list(value.std(0) if value.shape[0] != 1 else torch.zeros_like(value.mean(0))),
                    'max': self.to_list(value.max(0)[0]),
                    'min': self.to_list(value.min(0)[0]),
                }
        return result_dicts

    def display(self, result_dicts):
        table = Table('results')
        average, std = {}, {}
        for key in result_dicts.keys():
            value = ['{:.3f}'.format(v) for v in result_dicts[key][get_eval_fn_cmp(key)]]
            table.add_column(key, value)
            average[key] = '{:.3f}'.format(np.mean(result_dicts[key][get_eval_fn_cmp(key)]))
            std[key] = '{:.3f}'.format(np.std(result_dicts[key][get_eval_fn_cmp(key)]))
        # for average
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['mean' for _ in result_dicts.keys()])
        table.add_row(average.values())
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['std' for _ in result_dicts.keys()])
        table.add_row(std.values())

        return table.get_string()

    def log_wandb(self, result_dicts, batch_size):
        for s in range(batch_size):
            log_dict = {key: result_dicts[key][get_eval_fn_cmp(key)][s] for key in result_dicts.keys()}
            wandb.log(log_dict)
        log_dict = {key: np.mean(result_dicts[key][get_eval_fn_cmp(key)]) for key in result_dicts.keys()}
        new_log_dict = {key + '_all': value for key, value in log_dict.items()}
        wandb.log(new_log_dict)
        return


class Table(object):
    def __init__(self, title=None, field_names=None):
        """
            title:          str
            field_names:    list of field names
        """
        self.table = prettytable.PrettyTable(title=title, field_names=field_names)

    def add_rows(self, rows):
        """
            rows: list of tuples
        """
        self.table.add_rows(rows)

    def add_row(self, row):
        self.table.add_row(row)

    def add_column(self, fieldname, column):
        self.table.add_column(fieldname=fieldname, column=column)

    def get_string(self):
        """
            a markdown format table
        """
        _junc = self.table.junction_char
        if _junc != "|":
            self.table.junction_char = "|"
        markdown = [row for row in self.table.get_string().split("\n")[1:-1]]
        self.table.junction_char = _junc
        return "\n" + "\n".join(markdown)

    def get_latex_string(self):
        # TODO: to be done in future
        pass


__EVAL_FN__ = {}
__EVAL_FN_CMP__ = {}


def register_eval_fn(name: str):
    def wrapper(cls):
        if __EVAL_FN__.get(name, None):
            if __EVAL_FN__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __EVAL_FN__[name] = cls
        __EVAL_FN_CMP__[name] = cls.cmp
        cls.name = name
        return cls

    return wrapper


def get_eval_fn(name: str, **kwargs):
    if __EVAL_FN__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __EVAL_FN__[name](**kwargs)


def get_eval_fn_cmp(name: str):
    return __EVAL_FN_CMP__[name]


class EvalFn(ABC):
    def norm(self, x):
        return (x * 0.5 + 0.5).clip(0, 1)

    @abstractmethod
    def __call__(self, gt, measurement, sample, reduction='none'):
        pass


@register_eval_fn('psnr')
class PeakSignalNoiseRatio(EvalFn):
    cmp = 'max'  # the higher, the better

    def __call__(self, gt, measurement, sample, reduction='none'):
        return psnr(self.norm(gt), self.norm(sample), data_range=1.0, reduction=reduction)


@register_eval_fn('ssim')
class StructuralSimilarityIndexMeasure(EvalFn):
    cmp = 'max'  # the higher, the better

    def __call__(self, gt, measurement, sample, reduction='none'):
        return ssim(self.norm(gt), self.norm(sample), data_range=1.0, reduction=reduction)


@register_eval_fn('lpips')
class LearnedPerceptualImagePatchSimilarity(EvalFn):
    cmp = 'min'  # the higher, the better

    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.lpips_fn = LPIPS(replace_pooling=True, reduction='none')

    def evaluate_in_batch(self, gt, pred):
        batch_size = self.batch_size
        results = []
        for start in range(0, gt.shape[0], batch_size):
            res = self.lpips_fn(self.norm(gt[start:start+batch_size]), self.norm(pred[start:start+batch_size]))
            results.append(res)
        results = torch.cat(results, dim=0)
        return results

    def __call__(self, gt, measurement, sample, reduction='none'):
        res = self.evaluate_in_batch(gt, sample)
        if reduction == 'mean':
            res = res.mean()
        return res
    
    
path_to_generated = '/mntb/jkellerData/inverse-ipcn/'
path_to_true = '~/'

if __name__ == '__main__':
    # Your loading code
    # new_df = get_multi_paired_df('phase_retrieval')
    # ids_1 = new_df['alt_id_1'][:100].tolist()
    # ids_2 = new_df['alt_id_2'][:100].tolist()
    # ids_3 = new_df['alt_id_3'][:100].tolist()
    
    # generated_tensors = load_tensors_by_id(path_to_generated, ids_2)

    # processed_generated_tensors = torch.stack([torch.from_numpy(normalize_tensor(tensor)).permute(2, 0, 1) for tensor in generated_tensors])
    # true_tensors = load_true_images(100)
    
    true_tensors = load_true_tensors_by_id()


    # generated_tensors = load_local_tensors_by_id('phase_retrieval', 124, run=2)
    # processed_generated_tensors = torch.stack([torch.from_numpy(normalize_tensor(tensor)).permute(2, 0, 1) for tensor in generated_tensors])
    generated_tensors = load_local_tensors('phase_retrieval', 31)
    processed_generated_tensors = torch.stack([torch.from_numpy(normalize_tensor(tensor)).permute(2, 0, 1) for tensor in generated_tensors])

    # # # Initialize and Run
    evaluator = Evaluator([
        get_eval_fn('psnr'),
        get_eval_fn('ssim'),
        get_eval_fn('lpips')
    ])

    # Generate stats and print table
    with torch.no_grad():
        stats = evaluator.report(true_tensors, true_tensors, processed_generated_tensors)
    print(evaluator.display(stats))

# |  mean  |  mean |  mean |
# | 32.743 | 0.879 | 0.171 |
# |        |       |       |
# |  std   |  std  |  std  |
# | 9.813  | 0.154 | 0.224 |



# psnr_data = {"m": [33.994 , 32.743, 32.206], "s": [9.033 ,9.813, 10.067]}
# ssim_data = {"m": [0.894, 0.879, 0.860], "s": [0.148, 0.154, 0.163]}
# lpips_data = {"m": [0.145,   0.171, 0.198], "s": [0.210, 0.224, 0.236]}



# psnr_data = {"m": [29.998, 30.426, 30.217], "s": [3.798, 3.898, 3.991]}
# ssim_data = {"m": [0.953, 0.954, 0.953], "s": [0.025, 0.024, 0.024]}
# lpips_data = {"m": [0.102, 0.101, 0.101], "s": [0.057, 0.055, 0.056]}


# import numpy as np

# def pooled_stats(data):
#     m = np.array(data["m"])
#     s = np.array(data["s"])
    
#     mu = m.mean()
#     var_within = np.mean(s**2)
#     var_between = np.mean((m - mu)**2)
#     sigma = np.sqrt(var_within + var_between)
    
#     return mu, sigma

# for name, d in [("PSNR", psnr_data), ("SSIM", ssim_data), ("LPIPS", lpips_data)]:
#     mu, std = pooled_stats(d)
#     print(f"{name}: {mu:.3f} ± {std:.3f}")