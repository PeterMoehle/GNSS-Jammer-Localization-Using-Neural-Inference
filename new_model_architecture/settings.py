"""
Makes sure all the settings are loaded and exist. to override values, please make your own config.env file
"""

from pydantic_settings.main import SettingsConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingSettings(BaseSettings):
    BATCH_SIZE: int = 8
    LEARNING_RATE: float = 0.0006703373871168542
    WEIGHT_DECAY: float = 0.00001
    NUM_EPOCHS: int = 50
    VAL_DISCRITE_COEFF: float = 0.1
    TEST_DISCRITE_COEFF: float = 0.1
    RUN_PATH: str = "runs"

class ModelArchitectureSettings(BaseSettings):
    MODEL_TYPE: str = "GAT"
    IN_CHANNELS: int = 7
    OUT_CHANNELS: int = 128
    HIDDEN_CHANNELS: int = 128
    NUM_LAYERS: int = 8
    NUM_HEADS: int = 4
    OUT_FEATURES: int = 3
    DROPOUT_RATE: float = 0.0

class ValTestSettings(BaseSettings):
    TEST_BATCH_SIZE: int = 8
    RESUME: bool = False
    INFERENCE: bool = False 

class Settings(
    TrainingSettings,
    ModelArchitectureSettings,
    ValTestSettings
):
    model_config = SettingsConfigDict(env_file="config.env")


settings: Settings = Settings()