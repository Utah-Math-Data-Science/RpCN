from losses import  backward_loss
import torch
import tqdm as TQDM
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


class Trainer:
    def __init__(self, model, source_dist, target_dist, batch_size):
        self.model = model.to(device)  
        self.source = source_dist
        self.target = target_dist
        self.batch_size = batch_size

    def sample_t_and_r(self, n):
        s = torch.rand(n, 2, device=device)
        t = torch.max(s[:, 0], s[:, 1]).unsqueeze(1)
        r = torch.min(s[:, 0], s[:, 1]).unsqueeze(1)
        return t, r
    
    def calc_loss_batch(self, epoch):

        src = self.source.generate(self.batch_size).to(device)
        tgt = self.target.generate(self.batch_size).to(device)
        t, r = self.sample_t_and_r(self.batch_size)
        
        return backward_loss(self.model, src, tgt, t, r)
        

    
    def train_mean_flow_model(self, num_batches_per_epoch, n_epochs,
                            lr):
        opt = optim.AdamW(self.model.parameters(), lr=lr)

        pbar = TQDM.tqdm(range(n_epochs), desc=f"Training")
        
        # Initialize tracking dictionaries

        for epoch in pbar:
            epoch_loss = 0.0
            # Batch metrics for this epoch
            for batch_idx in range(num_batches_per_epoch):
                loss = self.calc_loss_batch(epoch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches_per_epoch

            pbar.set_description(f"Epoch {epoch} Loss {avg_loss:.4f} ")
        
        return self.model.to('cpu')