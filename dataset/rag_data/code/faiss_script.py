import faiss
import torch
import numpy as np

# Assume you have a dataset of past windows stored as a tensor (N, 36)
historical_windows = torch.randn(10000, 36)  # Example: 10K windows

# Convert to FAISS format
index = faiss.IndexFlatL2(36)  # L2 similarity (Euclidean distance)
index.add(historical_windows.numpy().astype(np.float32))  # Add to FAISS DB




batch_windows = torch.randn(20, 36) 



batch_size = batch_windows.shape[0]

# Convert to numpy for FAISS query
batch_np = batch_windows.numpy().astype(np.float32)

# Query FAISS for top-5 similar windows
_, retrieved_idx = index.search(batch_np, 5)  # (B, 5) indices
print("retrieved_idx",retrieved_idx)

# Fetch the retrieved windows
retrieved_windows = historical_windows[torch.tensor(retrieved_idx)]  # Shape: [B, 5, 36]
result = torch.tensor(retrieved_windows, dtype=torch.float32).to(batch_windows.device)
#print("result.shape",result.shape)
#print("result",result)
print("--------------------------------------------")

index.remove_ids(np.arange(len(historical_windows)))


# Query FAISS for top-5 similar windows
_, retrieved_idx = index.search(batch_np, 5)  # (B, 5) indices
print("retrieved_idx",retrieved_idx)

# Fetch the retrieved windows
retrieved_windows = historical_windows[torch.tensor(retrieved_idx)]  # Shape: [B, 5, 36]
result = torch.tensor(retrieved_windows, dtype=torch.float32).to(batch_windows.device)
print("result.shape",result.shape)
print("result",result)



# Assume new_windows is a tensor of new time-series data
new_windows = torch.randn(2000, 36)

# Add new embeddings
index.add(new_windows)




