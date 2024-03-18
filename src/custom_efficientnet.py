import pytorch_lightning as pl
import torch
import torch.nn as nn

from config import *
from torch.optim import Adam
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class MyEfficientNet(pl.LightningModule):
    
    def __init__(self, 
             num_classes,
             batch_size,
             learning_rate,
             fine_tune=True,
             class_weights=None
            ):
        
        super(MyEfficientNet, self).__init__()
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Initialize loss function (optional: add class weights)s
        if class_weights is not None:
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss = nn.CrossEntropyLoss()

        # Load Model with pre-trained weights
        self.efficientnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

        # Add final layer to predict 2 classes
        if num_classes is not None:
            self.efficientnet.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
        else:
            # Remove final layer
            self.efficientnet = torch.nn.Sequential(*(list(self.efficientnet.children())[:-1]))

        # Freeze or unfreeze layers
        for param in self.efficientnet.parameters():
            param.requires_grad = fine_tune

    
    def forward(self, x):
        return self.efficientnet(x)

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    
    def common_step(self, batch, batch_idx, batch_size):
        # Extract data from batch
        data = batch[0]["data"]
        # Remove sequence length dim
        data = torch.squeeze(data, 1).reshape(batch_size,3, 189, 224)
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

    def predict_step(self, batch, batch_idx):
        # Extract data from batch
        data = batch[0]["data"]
        
        # Remove sequence length dim
        data = torch.squeeze(data, 1).reshape(batch_size,3, 189, 224)

        # Make prediction
        outputs = self.forward(data)

        return outputs