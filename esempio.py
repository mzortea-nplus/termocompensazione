from src.config import load_config
from src.data import load_data

cfg = load_config("config.yaml")

df, tmp_sensors, sensors = load_data(cfg)

