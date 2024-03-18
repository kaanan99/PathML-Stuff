import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from config import *
from custom_efficientnet import MyEfficientNet
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.optim import Adam

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


class LRCN(pl.LightningModule):
    def __init__(
            self, 
            hidden_size, 
            num_classes, 
            num_stacked_layers, 
            device, 
            learning_rate=.00001, 
            batch_size=BATCH_SIZE, 
            sequence_length=SEQUENCE_LENGTH, 
            fine_tune_efficientnet=True,
            custom_efficientnet=False,
            checkpoints=None,
            class_weights=None,
        ):
        
        super(LRCN, self).__init__()
        
        # Metadata
        if class_weights is not None:
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.device_ = device
        
        # Information about data
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        # EfficientNet with final layer dropped (output of 1280)
        if not custom_efficientnet:
            self.efficientnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            self.efficientnet = torch.nn.Sequential(*(list(self.efficientnet.children())[:-1]))
        else:
            # Load checkpoints
            if checkpoints:
                self.efficientnet = MyEfficientNet.load_from_checkpoint(
                    checkpoints,  
                    num_classes=num_classes,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    fine_tune=fine_tune_efficientnet,
                    class_weights=None, # Revert back to class_weights 
                )
            else:
                self.efficientnet = MyEfficientNet(
                    num_classes,
                    self.batch_size,
                    self.learning_rate,
                    fine_tune_efficientnet,
                    class_weights,
                )
            self.efficientnet = torch.nn.Sequential(*(list(list(self.efficientnet.children())[1].children())[:-1]))
        
        # Initialize outputsize
        self.efficientnet_output_size = 1280

        # Remove final layer
        # self.efficientnet = torch.nn.Sequential(*(list(self.efficientnet.children())[:-1]))

        # Freeze or unfreeze layers
        for param in self.efficientnet.parameters():
                param.requires_grad = fine_tune_efficientnet
        
        if not fine_tune_efficientnet:
            self.efficientnet.eval()

        
        # LSTM
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(
            self.efficientnet_output_size, 
            self.hidden_size, 
            self.num_stacked_layers, 
            batch_first=True
        )

        # Final layer for determining class
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        current_batch_size = x.shape[0]
        
        # Get a list of features for each image in the sequence
        efficient_net_output = [self.efficientnet(image) for image in x]        
        # Convert into a single tensor of dimension (Batch size, sequence length, 1280)
        efficient_net_output = (
            torch.stack(efficient_net_output)
            .reshape(current_batch_size, self.sequence_length, self.efficientnet_output_size)
        )

        # Initialize blank gates
        h_0 = torch.zeros(self.num_stacked_layers, current_batch_size, self.hidden_size).to(self.device_)
        c_0 = torch.zeros(self.num_stacked_layers, current_batch_size, self.hidden_size).to(self.device_)

        # LSTM output
        lstm_out, _ = self.lstm(efficient_net_output, (h_0, c_0))
        out = self.fc(lstm_out[:, -1, :])
        return out

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    
    def common_step(self, batch, batch_idx, batch_size):
        # Extract data from batch
        data = batch[0]["data"].reshape(batch_size, SEQUENCE_LENGTH, 3, 189, 224)
        labels = batch[0]["label"]

        # Forward prop
        outputs = self.forward(data)
        loss = self.loss(outputs, labels)

        #Evaluation
        predicted_classes = torch.argmax(outputs, dim=1).detach().cpu()
        actual_classes = torch.argmax(labels, dim=1).detach().cpu()
        correct = (predicted_classes == actual_classes).sum().item()
        accuracy = correct / self.batch_size
        return loss, accuracy
        
    
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx, self.batch_size)
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)
        return loss

    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx, VAL_BATCH_SIZE)
        self.log("testing_loss", loss)
        self.log("testing_accuracy", accuracy)
        return loss

    
    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx, VAL_BATCH_SIZE)
        self.log("eval_loss", loss)
        self.log("eval_accuracy", accuracy)
        return loss
