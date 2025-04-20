# train.py
import os
import torch
import random
from torch.utils.data import DataLoader
from params import par
# adjust these imports to match your Dataloader_loss.py
from Dataloader_loss import OdometryDataset  # or SequenceDataset
from models.deepvo import DeepVO

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model):
    opt = par.optim.get('opt', 'Adam')
    lr  = par.optim.get('lr', 1e-3)
    if opt == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    elif opt == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif opt == 'Cosine':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=par.optim.get('T', 100))
        return optimizer, scheduler
    else:
        return torch.optim.Adam(model.parameters(), lr=lr)


def train():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets & Loaders
    from torch.utils.data import ConcatDataset

    # Build training dataset by concatenating per-sequence loaders
    seq_len = par.seq_len[0]  # use fixed seq_len
    train_dsets = [OdometryDataset(seq, seq_len) for seq in par.train_video]
    train_dataset = ConcatDataset(train_dsets)
    # Instead of:
    # valid_dataset = OdometryDataset(par.valid_data_info_path)
    # Now:
    valid_dsets = [OdometryDataset(seq, seq_len) for seq in par.valid_video]
    valid_dataset = ConcatDataset(valid_dsets)
    train_loader = DataLoader(train_dataset, batch_size=par.batch_size, shuffle=True,
                              num_workers=par.n_processors, pin_memory=par.pin_mem)
    valid_loader = DataLoader(valid_dataset, batch_size=par.batch_size, shuffle=False,
                              num_workers=par.n_processors, pin_memory=par.pin_mem)

    # Model, Optimizer, Loss
    model = DeepVO(feature_dim=512, rnn_hidden=par.rnn_hidden_size).to(device)
    optimizer = get_optimizer(model)
    scheduler = None
    if isinstance(optimizer, tuple):
        optimizer, scheduler = optimizer
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(1, par.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for images, poses in train_loader:
            images = images.to(device)         # (B, T, C, H, W)
            poses  = poses.to(device)          # (B, T, 15)
            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, poses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, poses in valid_loader:
                images = images.to(device)
                poses  = poses.to(device)
                preds  = model(images)
                loss   = criterion(preds, poses)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(valid_dataset)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Scheduler step (if any)
        if scheduler is not None:
            scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(par.save_model_path), exist_ok=True)
            torch.save(model.state_dict(), par.save_model_path)
            print(f"Saved best model (val_loss={val_loss:.4f}) at epoch {epoch}")

if __name__ == '__main__':
    train()
