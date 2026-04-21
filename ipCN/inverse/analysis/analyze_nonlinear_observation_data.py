import torch 
from torchmetrics.aggregation import MeanMetric, SumMetric

def create_metric_state(means, stds, n=100):
    """
    Creates torchmetrics objects and populates them with provided summary stats.
    """
    mean_tracker = MeanMetric()
    sq_sum_tracker = SumMetric()

    for m, s in zip(means, stds):
        # Convert to tensors
        m_t = torch.tensor(float(m))
        s_t = torch.tensor(float(s))
        
        # 1. Update Mean tracker (weighted by n)
        # .update() expects the values themselves, but we can simulate n updates
        # by passing the mean with a weight of n.
        mean_tracker.update(m_t, weight=n)
        
        # 2. Update Squared Sum tracker
        # Formula: Sum(x^2) = n * (std^2 + mean^2)
        total_sq_sum = n * (s_t**2 + m_t**2)
        sq_sum_tracker.update(total_sq_sum)
        
    return mean_tracker, sq_sum_tracker

def compute_final(mean_tracker, sq_sum_tracker, total_n=300):
    mu = mean_tracker.compute()
    # E[X^2] - (E[X])^2
    var = (sq_sum_tracker.compute() / total_n) - mu**2
    return mu.item(), torch.sqrt(var).item()



psnr_data = {"m": [29.998, 30.426, 30.217], "s": [3.798, 3.898, 3.991]}
ssim_data = {"m": [0.953, 0.954, 0.953], "s": [0.025, 0.024, 0.024]}
lpips_data = {"m": [0.102, 0.101, 0.101], "s": [0.057, 0.055, 0.056]}


metrics = {"PSNR": psnr_data, "SSIM": ssim_data, "LPIPS": lpips_data}

print(f"{'Metric':<10} | {'Mean ± Std':<20}")
print("-" * 35)

for name, data in metrics.items():
    m_track, s_track = create_metric_state(data['m'], data['s'])
    final_m, final_s = compute_final(m_track, s_track)
    print(f"{name:<10} | {final_m:>7.3f} ± {final_s:<7.3f}")