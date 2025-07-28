import os
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import StandardScaler

# Define parameters
folder_path = "/scratch/fs47816/workdir/sample_scripts/time_series_dl/time-series-v5/Time-Series-Library/dataset/rag_data/data"  # Change to your folder path
window_size = 36
window_stride = 1

# Initialize list to store all normalized windows
all_windows = []

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Ignore the first column (date column)
        data = df.iloc[:, 1:].values  # Extract numerical data only
        
        # Process each column (time series)
        for col in range(data.shape[1]):
            series = data[:, col]  # Extract time series
            
            # Generate sliding windows
            num_windows = len(series) - window_size + 1
            if num_windows <= 0:
                continue  # Skip if not enough data for at least one window
            
            windows = [series[i : i + window_size] for i in range(0, num_windows, window_stride)] #np.array([series[i : i + window_size] for i in range(0, num_windows, window_stride)])
            
            # Normalize each window using StandardScaler
            scaler = StandardScaler()
            normalized_windows = normalized_windows = np.array([StandardScaler().fit_transform(w.reshape(-1, 1)).flatten() for w in windows]) #scaler.fit_transform(windows)  # Standardize each window
            
            # Store in list
            all_windows.append(normalized_windows)

# Convert to NumPy array
all_windows = np.vstack(all_windows).astype(np.float32)  # Shape (N, 36), where N is total windows

# Initialize Faiss GPU index (L2 similarity)
cpu_index = faiss.IndexFlatL2(window_size)  # Create CPU index
gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)  # Move index to GPU 

# Add data to GPU Faiss index
gpu_index.add(all_windows)

# Print index details
print(f"Faiss GPU index contains {gpu_index.ntotal} vectors of size {window_size}.")

faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), "index_faiss.bin")




