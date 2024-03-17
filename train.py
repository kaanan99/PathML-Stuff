import numpy as np
import os
import pytorch_lightning as pl
import torch

from config import *
from data_module import VideoDataModule
from lrcn import LRCN
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from tqdm import tqdm

from sklearn.metrics import(
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
)


def evaluate_model(model, data, device, val=True):
    if val:
        batch_size = VAL_BATCH_SIZE
    else:
        batch_size = BATCH_SIZE
    
    # Set model to eval and put on device
    model.eval()
    model.to(device)

    predicted_label = []
    actual_label = []

    classes = [NUM_TO_LABEL[i] for i in range(NUM_CLASSES)]

    for batch in tqdm(data):
        # Get Data
        data = batch[0]["data"].reshape(batch_size, SEQUENCE_LENGTH, 3, 189, 224)
        label = batch[0]["label"]

        # Make Predictions
        outputs = model(data)

        # Decode prediction from one hot encoding ([0, 1, 0] --> 1)
        predicted_class = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        actual_class = torch.argmax(label, dim=1).detach().cpu().numpy()

        # Save Results
        predicted_label.append(predicted_class)
        actual_label.append(actual_class)

    # Flatten lists (num_batches, batch_size) -> (batch_size * num_batches)
    actual_label = np.array(actual_label).flatten()
    predicted_label = np.array(predicted_label).flatten()

    # Create confusion matrix
    cm_train = confusion_matrix(actual_label, predicted_label)
    confusion_matrix_object = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=classes)
    confusion_matrix_object.plot()

    # Calculate metrics
    accuracy = accuracy_score(actual_label, predicted_label)

    # Display metrics
    print(f"Accuracy: {accuracy}")
    
    return confusion_matrix_object


def calculate_class_weights(loader, device):
    unique_classes = list(NUM_TO_LABEL.keys())
    total_labels = []
    
    for batch in tqdm(loader):
        data = batch[0]["label"]
        batch_labels = torch.argmax(data, dim=1).detach().cpu().numpy()
        total_labels.extend(batch_labels)
    
    weights = class_weight.compute_class_weight(
        "balanced", 
        classes=unique_classes, 
        y=total_labels
    )
    class_weights = torch.FloatTensor(weights).to(device)
    return class_weights


def main():
    
    """
    -----------------------------------------
    SETUP
    -----------------------------------------
    """

    # Determine Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using: ", device)
    torch.cuda.empty_cache()

    # Create data module
    video_data_module = VideoDataModule(
        BATCH_SIZE, 
        TRAIN_FILE_LIST, 
        TEST_FILE_LIST, 
        EVAL_FILE_LIST, 
        TRAIN_SIZE, 
        TEST_SIZE, 
        EVAL_SIZE
    )

    # Get class weights
    if CLASS_WEIGHTS:
        print("\nCalculating class weights:")
        training_loader = video_data_module.train_dataloader()
        training_class_weights = calculate_class_weights(training_loader, device)
    else:
        training_class_weights = None

    print("\nBeginning Setup")
    
    # Create logger
    logger = TensorBoardLogger(
        "tensor_board_logs", 
        name=EXPERIMENT_NAME,
    )

    # Create checkpoint callback (Saves model state)
    # Saves checkpoints every epoch
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./checkpoints/{EXPERIMENT_NAME}', 
        save_on_train_epoch_end=True,
        monitor='testing_accuracy',
        mode='max',
        filename="{EXPERIMENT_NAME}_epoch_{epoch}_val_accuracy_{testing_accuracy:.2f}_step_{step}",
        save_top_k=-1, # Save all models 
        every_n_epochs=1,
    )

    # Runs Val every epoch
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1, 
        max_epochs=EPOCHS,
        default_root_dir=os.getcwd(),
        callbacks=[checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=5,
        fast_dev_run=FAST_DEV_RUN
    )
    
    # Create model
    lrcn = LRCN(
        HIDDEN_SIZE, 
        NUM_CLASSES, 
        NUM_HIDDEN_LAYERS, 
        fine_tune_efficientnet=False, 
        device=device,
        custom_efficientnet=True,
        checkpoints="final_checkpoints/EfficientNet_trial_1/final_checkpoint",
        class_weights=training_class_weights,
    ).to(device)

    print("\nFinish Setup")
    """
    -----------------------------------------
    TRAINING
    -----------------------------------------
    """

    print("\n Beginning Training") 
    
    # Train model
    trainer.fit(lrcn, video_data_module)

    # Save final checkpoint
    trainer.save_checkpoint(f"./final_checkpoints/{EXPERIMENT_NAME}/final_checkpoint")

    # Get validation results
    trainer.validate(lrcn, video_data_module)


    """
    -----------------------------------------
    EVALUATION
    -----------------------------------------
    """
    print("\nEvaluating Validation set")

    # Extract validation dataloader from data module
    val_data = video_data_module.val_dataloader()


    # Get confusion matrix and metrics
    confusion_matrix = evaluate_model(lrcn, val_data, device)
    confusion_matrix.figure_.savefig(f'./confusion_matrices/{EXPERIMENT_NAME}_validation_confusion_matrix.png')

    print("\nEvaluating Training set")
    
    # Extract training dataloader from data module
    train_data = video_data_module.train_dataloader()
    
    # Get confusion matrix and metrics
    confusion_matrix_ = evaluate_model(lrcn, train_data, device, val=False)
    confusion_matrix_.figure_.savefig(f'./confusion_matrices/{experiment_name}_training_confusion_matrix.png')
    


if __name__ == "__main__":
    main()
