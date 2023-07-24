import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

probscore = pd.read_csv("score.csv", sep="\s+", header=0)
print(probscore.keys())
print(probscore.head())


humaninfo = pd.read_csv("human50_1000info.csv", sep="\t",header=0, index_col=0)
print(humaninfo.keys())
print(humaninfo.head())


humanscore = pd.merge(probscore, humaninfo, how='left', left_on=['Ensemble-id'], right_on = ['ENSPdotID'])
humanscore.sort_values(by='positive', ascending=False, inplace=True)
humanscore.to_csv('humanrankedscore.csv', sep="\t")

undrug = pd.read_csv('finalundrug.txt', header=0, index_col=False)
print(undrug.shape)
print(undrug.keys())
print(undrug.head())

scoreinundrug = pd.merge(humanscore, undrug, how='inner', left_on=['Uniprot'], right_on = ['uniprot'])
print(scoreinundrug.shape)
print(scoreinundrug.keys())
print(scoreinundrug.head())

cutoff067 = scoreinundrug[scoreinundrug['positive']  > 0.67]
cutoff067.to_csv('067cutoff.csv', sep="\t")
cutoff067uniprot = cutoff067[['Uniprot']].drop_duplicates()
cutoff067uniprot.to_csv('unip067cutoff.csv', sep="\t")

cutoff070 = scoreinundrug[scoreinundrug['positive']  > 0.70]
cutoff070.to_csv('070cutoff.csv', sep="\t")
cutoff070uniprot = cutoff070[['Uniprot']].drop_duplicates()
cutoff070uniprot.to_csv('unip070cutoff.csv', sep="\t")

cutoff075 = scoreinundrug[scoreinundrug['positive']  > 0.75]
cutoff075.to_csv('075cutoff.csv', sep="\t")
cutoff075uniprot = cutoff075[['Uniprot']].drop_duplicates()
cutoff075uniprot.to_csv('unip075cutoff.csv', sep="\t")

cutoff080 = scoreinundrug[scoreinundrug['positive']  > 0.80]
cutoff080.to_csv('080cutoff.csv', sep="\t")
cutoff080uniprot = cutoff080[['Uniprot']].drop_duplicates()
cutoff080uniprot.to_csv('unip080cutoff.csv', sep="\t")

cutoff085 = scoreinundrug[scoreinundrug['positive']  > 0.85]
cutoff085.to_csv('085cutoff.csv', sep="\t")
cutoff085uniprot = cutoff085[['Uniprot']].drop_duplicates()
cutoff085uniprot.to_csv('unip085cutoff.csv', sep="\t")

cutoff090 = scoreinundrug[scoreinundrug['positive']  > 0.90]
cutoff090.to_csv('090cutoff.csv', sep="\t")
cutoff090uniprot = cutoff090[['Uniprot']].drop_duplicates()
cutoff090uniprot.to_csv('unip090cutoff.csv', sep="\t")

cutoff095 = scoreinundrug[scoreinundrug['positive']  > 0.95]
cutoff095.to_csv('095cutoff.csv', sep="\t")
cutoff095uniprot = cutoff095[['Uniprot']].drop_duplicates()
cutoff095uniprot.to_csv('unip095cutoff.csv', sep="\t")

