from src.model import ANN
from src.prepare_dataloader import DiabetesDataLoader
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

class Trainer:
    
    def __init__(self):
        
        self.model = ANN()
        data_loader = DiabetesDataLoader()
        
        self.train_loader, self.valid_loader = data_loader.create_dataloder()
    
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = 1e-03
        )
    
    def train(self, epochs: int =5):
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            counter = 0
            train_accuracy_score = 0
            for i, batch in enumerate(self.train_loader):
                counter += 1
                self.optimizer.zero_grad()
                
                pred = self.model(
                    batch["features"]
                )
                
                loss = self.criterion(pred, batch["label"])
                
                epoch_loss += loss.item()
                train_accuracy_score += accuracy_score(
                    y_true = batch["label"].tolist(), 
                    y_pred = pred.argmax(axis=1).tolist())
                
                loss.backward()
                self.optimizer.step()

            print(f"epoch: {epoch+1} with training loss : {epoch_loss/counter} and training accuracy: {train_accuracy_score/counter}")
            
            self.model.eval()
            eval_loss = 0
            eval_counter = 0
            eval_accuracy_score = 0
            with torch.no_grad():
                for i, batch in enumerate(self.valid_loader):
                    
                    eval_counter += 1
                    valid_pred = self.model(
                        batch["features"]
                    )
                    
                    loss = self.criterion(valid_pred, batch["label"])
                    eval_loss += loss.item()
                    
                    eval_accuracy_score += accuracy_score(
                        y_true = batch["label"].tolist(), 
                        y_pred = valid_pred.argmax(axis=1).tolist())
                
            print(f"epoch: {epoch+1} with validation loss : {eval_loss/eval_counter} and validation accuracy score: {eval_accuracy_score/eval_counter}")
            
                
                
if __name__ == "__main__":
    
    trainer = Trainer()
    trainer.train() 
    print(torch.__version__)
