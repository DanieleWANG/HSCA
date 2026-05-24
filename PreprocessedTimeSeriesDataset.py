
import pickle
import numpy as np
import pandas as pd


class PreprocessedTimeSeriesDataset:
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.data = data_dict['data']  # (N, T, F) numpy array
        self.index = data_dict['index']  # pd.MultiIndex
        self.length = len(self.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self.data[idx]  # (T, F)
        elif isinstance(idx, (list, np.ndarray, tuple)):
            return self.data[idx]  # (N, T, F)
        else:
            return self.data[int(idx)]
    
    def get_index(self):
        return self.index


