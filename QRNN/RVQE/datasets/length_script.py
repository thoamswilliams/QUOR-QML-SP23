import pandas as pd
import os.path as path

DATASET_PATH_TRAIN = path.join(path.dirname(path.abspath(__file__)), "res/twitter_less_than_8.csv")


df = pd.read_csv(DATASET_PATH_TRAIN, delimiter = ',')
input_text = df["Tweets"].tolist()
labels = df["Score"].tolist()

print(input_text)
print(labels)
print(max([len(i) for i in input_text]))