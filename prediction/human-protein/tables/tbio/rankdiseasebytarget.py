import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

cut085probscore = pd.read_csv('085cutoff.csv', sep="\t", header=0, index_col=0)
print(cut085probscore.shape)
print(cut085probscore.keys())
print(cut085probscore.head())


cut085uniprot = cut085probscore[['uniprot']].drop_duplicates()

unip2geneid = pd.read_csv('mapa_geneid_4_uniprot_crossref.tsv', sep="\t", header=0, index_col=False, engine='python')

print(unip2geneid.shape)
print(unip2geneid.keys())
print(unip2geneid.head())

cut085unipgeneid = pd.merge(cut085uniprot, unip2geneid, how='inner', left_on=['uniprot'], right_on = ['UniProtKB'])
print(cut085unipgeneid.shape)
print(cut085unipgeneid.keys())
print(cut085unipgeneid.head())

geneid2diseasetmp = pd.read_csv('all_gene_disease_associations.tsv', sep="\t", header=0, index_col=False)

geneid2disease = geneid2diseasetmp[['geneId','diseaseName','diseaseSemanticType']]

print(geneid2disease.shape)
print(geneid2disease.keys())
print(geneid2disease.head())

cut085diseasetmp = pd.merge(cut085unipgeneid, geneid2disease, how='inner', left_on=['GENEID'], right_on = ['geneId'])

cut085disease = cut085diseasetmp[['uniprot','GENEID','diseaseName','diseaseSemanticType']]

print(cut085disease.shape)
print(cut085disease.keys())
print(cut085disease.head())


cut085disease.to_csv('cut085disease.csv')

rankdisease = cut085disease[['GENEID','diseaseName']].groupby('diseaseName').agg('count').sort_values('GENEID',ascending=False)

rankdisease.reset_index()
rankdisease.to_csv('rankeddisease.csv')

ranksemantic = cut085disease[['GENEID','diseaseSemanticType']].groupby('diseaseSemanticType').agg('count').sort_values('GENEID',ascending=False)

ranksemantic.reset_index()
ranksemantic.to_csv('rankedsemantic.csv')

