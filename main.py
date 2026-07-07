import kagglehub
import pandas as pd
import os

# Veri setini indir
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Path to dataset files:", path)
print(os.listdir(path))