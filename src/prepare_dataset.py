import pandas as pd 
from pandas import DataFrame
from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader, Subset
import torch

class DiabetesDataset:
    
    def __init__(self, file_path: str) -> None:
        
        self.file_path = file_path
        self.dataset = pd.read_csv(self.file_path)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        df = self.dataset.loc[idx]
        features = df.values[:-1]
        label = int(df.values[-1])
        
        # print(label)
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label, dtype = torch.long)
        }
    
if __name__ == "__main__":
    
    dataset = DiabetesDataset(file_path= "diabetes.csv")
    
    print(len(dataset))
    print(dataset[0])
    
    # valid_ratio = 0.1
    # valid_size = int(len(dataset) * valid_ratio)
    # # print(valid_size)
    
    # indices = torch.randperm(len(dataset)).tolist()
    # train_set = Subset(dataset, indices[:-valid_size])
    # valid_set = Subset(dataset, indices[-valid_size:])
    
    # train_loader = DataLoader(
    #         train_set,
    #         batch_size = 4,
    #         shuffle = True
    #     )
    
    # valid_loader = DataLoader(
    #         valid_set,
    #         batch_size = 4,
    #         shuffle = True
    #     )
    
    
    
    
    
        
        