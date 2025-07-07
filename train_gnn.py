import json
import torch
import torch.nn as nn
import torch.optim as optim
from model_gnn import GNNRoutePlanner
from utils_gnn import convert_to_gnn_input
import os

with open("prob1.json", "r") as f:
    prob_info = json.load(f)

x, edge_index = convert_to_gnn_input(prob_info)

N = x.shape[0]
y = torch.zeros(N)

for (src_dst, demand) in prob_info["K"]:
    src, dst = src_dst
    y[src] = 1
    y[dst] = 1

model = GNNRoutePlanner(in_channels=x.shape[1], hidden_channels=64, out_channels=1, activation="leaky_relu")
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("GNN training started")

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(x, edge_index).squeeze()

    print(f"[Epoch {epoch}] forward shape: {out.shape}")
    
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

os.makedirs("checkpoint", exist_ok=True)
torch.save(model.state_dict(), "checkpoint/model.pt")
print("Model saved to checkpoint/model.pt")
