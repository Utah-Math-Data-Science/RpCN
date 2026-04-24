import torch 
from loss import diffusion_loss
import torch.optim as optim
import tqdm as TQDM
device = 'cuda'



class DiffusionTrainer:
    def __init__(self, model, source_dist, target_dist, batch_size, alphas_cumprod):
        self.model = model.to(device)  # move once
        self.source = source_dist
        self.target = target_dist
        self.batch_size = batch_size
        self.alphas_cumprod = alphas_cumprod

    def sample_t(self, n):
        t = torch.rand(n, 1, device=device)

        return t
    
    def calc_loss_batch(self, epoch):
        # generate on CPU then move once to device (your generator may already provide tensors)
        src = self.source.generate(self.batch_size).to(device)
        tgt = self.target.generate(self.batch_size).to(device)
        t = self.sample_t(self.batch_size)
        
        return diffusion_loss(self.model, tgt, t, self.alphas_cumprod)        

    
    def train_diffusion_model(self, num_batches_per_epoch, n_epochs,
                            lr):
        opt = optim.AdamW(self.model.parameters(), lr=lr)

        pbar = TQDM.tqdm(range(n_epochs))
        
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

            
            # Store average metrics for this epoch
            avg_loss = epoch_loss / num_batches_per_epoch

            
            pbar.set_description(f" Epoch {epoch} Loss {avg_loss:.4f} ")
        
        return self.model.to('cpu')
