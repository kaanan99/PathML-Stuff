TASK = 1
NUM_CLASSES = 3
FPS = 1

TRAIN_FILE_LIST = f'/mnt/durable/training_data/filelists/task1_training.txt'
TEST_FILE_LIST = f'/mnt/durable/training_data/filelists/task1_testing.txt'
EVAL_FILE_LIST = f'/mnt/durable/training_data/filelists/task1_eval.txt'

TRAIN_SIZE =  350160 * FPS
TEST_SIZE  =   53642 * FPS
EVAL_SIZE  =   53942 * FPS

NUM_TO_LABEL = {
    0:"moving", 
    1: "sedentary", 
    2: "standing"
}

SEQUENCE_LENGTH = 1 # Change back for LCRN
BATCH_SIZE = 40
VAL_BATCH_SIZE = 10
EPOCHS = 15


HIDDEN_SIZE = 64
NUM_HIDDEN_LAYERS = 2

EXPERIMENT_NAME = "custom_efficientnet_classweights"

CLASS_WEIGHTS = True
FAST_DEV_RUN = False