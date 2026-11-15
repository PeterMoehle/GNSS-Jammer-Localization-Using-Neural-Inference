from typing import Any
MODEL_ARCHITECTURE_DEFAULTS: dict[str, Any] = {
    "MODEL_TYPE": "GAT",
    "IN_CHANNELS": 7,
    "OUT_CHANNELS": 128,
    "HIDDEN_CHANNELS": 128,
    "NUM_LAYERS": 8,
    "NUM_HEADS": 4,
    "OUT_FEATURES": 3,
    "DROPOUT_RATE": 0.0,
}

TRAINING_DEFAULTS: dict[str, Any] = {
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 0.0006703373871168542,
    "WEIGHT_DECAY": 0.00001,
    "NUM_EPOCHS": 2,
    "VAL_DISCRITE_COEFF": 0.1, #what
    "TEST_DISCRITE_COEFF": 0.1, #What
}

VAL_TEST_DEFAULTS: dict[str, Any] = {
    "TEST_BATCH_SIZE": 8,
    "RESUME": False, #For checkpointing
    "INFERENCE": False, #For inference
}