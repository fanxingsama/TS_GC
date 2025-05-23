# config.py
import torch

from data_loader import TimeSeriesDataloader


DATA_PATH = 'data\simu_data\series_data.csv'
gc_dir = 'data\simu_data\granger_causality.csv'
BATCH_SIZE = 128
DATA_SEED = 42
INPUT_WINDOW = 20
OUTPUT_WINDOW = 1
FEATURE_DIM = 1
OUTPUT_DIM = 1
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timeseriesDataLoader = TimeSeriesDataloader(data_dir=DATA_PATH, gc_dir=gc_dir, batch_size=BATCH_SIZE, 
                                            DATA_SEED=DATA_SEED, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW)
SERIES_NUM = timeseriesDataLoader.series_num