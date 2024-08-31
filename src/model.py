import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.metrics import f1_score

class ANN(nn.Module):
    
    def __init__(
        self,
        input_features: int=8,
        output_features: int =2
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(
            input_features,
            20
        )
        
        self.linear2 = nn.Linear(
            20,
            10
        )
        
        self.output = nn.Linear(
            10,
            output_features
        )
            
    def forward(self, x):
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x)
    
if __name__ == "__main__":
    
    from prepare_dataloader import DiabetesDataLoader
    
    data_loader = DiabetesDataLoader()
    train_loader , valid_loader = data_loader.create_dataloder()
    first_batch = next(iter(train_loader))
    
    model = ANN()
    # torch.squeeze(self.model(batch["text"].to(self.device)),-1)
    a = []
    out = model(first_batch["features"])
    print(first_batch["label"])
    print(out)   
    print(out.argmax(axis=1))    
    
    print(f1_score(first_batch["label"].tolist(),out.argmax(axis=1).tolist()))
    # criterion = nn.CrossEntropyLoss()
    
    # loss = criterion(out, first_batch["label"])
    # print(loss)
    # a= loss.item()
    # print(a)
    
    