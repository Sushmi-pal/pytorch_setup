from src.prepare_dataset import DiabetesDataset
from typing import Tuple
from torch.utils.data import Subset, DataLoader
import torch

class DiabetesDataLoader:
    
    def __init__(self, file_path = "data/diabetes.csv") -> None:
        
        self.dataset = DiabetesDataset(
            file_path = file_path
        )
        
    def split_train_valid(self, validation_ratio: float= 0.1) -> Tuple[DiabetesDataset, DiabetesDataset]:
        valid_size = int(len(self.dataset) * validation_ratio)
    
        indices = torch.randperm(len(self.dataset)).tolist()
        train_set = Subset(self.dataset, indices[:-valid_size])
        valid_set = Subset(self.dataset, indices[-valid_size:])
        
        return train_set, valid_set
    
    def create_dataloder(self, batch_size: int = 4)-> DataLoader:
        
        train_set, valid_set = self.split_train_valid()
        
        return DataLoader(train_set, batch_size=4, shuffle=True), DataLoader(valid_set, batch_size=4, shuffle=False)
    

if __name__ == "__main__":
    
    data_loader = DiabetesDataLoader()
    train_loader , valid_loader = data_loader.create_dataloder()
    first_batch = next(iter(train_loader))

    # Print the first item in the batch
    print(first_batch)
    
    first_batch = next(iter(valid_loader))

    # Print the first item in the batch
    print(first_batch) 
            