import pandas as pd


small_data = pd.read_json('../clickbait17-train-170331/instances.jsonl', lines= True, orient = 'columns')
small_truth = pd.read_json('../clickbait17-train-170331/truth.jsonl', lines= True, orient = 'columns')

big_data = pd.read_json('../clickbait17-train-170630/instances.jsonl', lines= True, orient = 'columns')
big_truth = pd.read_json('../clickbait17-train-170630/truth.jsonl', lines= True, orient = 'columns')

merged = pd.merge(small_data, small_truth, on='id',how='inner')
merged2 = pd.merge(big_data, big_truth, on='id',how='inner')

frames = [merged, merged2]
df = pd.concat(frames)

df = pd.read_csv("./integrated.csv")

