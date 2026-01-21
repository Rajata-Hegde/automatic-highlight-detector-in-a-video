import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import optim, nn
from model.autoencoder import AutoencoderRASL

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(feat_path, model_path='model/ae.pth', epochs=50, batch_size=32, lr=1e-3):
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        return

    print(f"🚀 Training Autoencoder on {feat_path}")
    
    data = np.load(feat_path)
    if data.ndim != 2:
        print("⚠️ Expected 2D feature array. Check shape.")
        return

    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoencoderRASL(input_dim=data.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"📘 Epoch [{epoch+1}/{epochs}] — Loss: {avg_loss:.6f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved → {model_path}")

if __name__ == "__main__":
    train(
        feat_path="features/myvideo.npy", 
        model_path="model/ae.pth", 
        epochs=50, 
        batch_size=32, 
        lr=1e-3
    )
