from src.config import load_config
from src.data import load_data

cfg = load_config("config.yaml")

df, tmp_sensors, sensors = load_data(cfg)
df = df.sort_values(by="time").reset_index(drop=True)
# print(df.head(-5))

print("sensors:", sensors)
    
