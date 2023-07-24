import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

cut090probscore = pd.read_csv('090cutoff.csv', sep="\t", header=0, index_col=0)
print(cut090probscore.shape)
print(cut090probscore.keys())
print(cut090probscore.head())


cut090uniprot = cut090probscore[['uniprot']].drop_duplicates()

unip2geneid = pd.read_csv('mapa_geneid_4_uniprot_crossref.tsv', sep="\t", header=0, index_col=False, engine='python')

print(unip2geneid.shape)
print(unip2geneid.keys())
print(unip2geneid.head())

cut090unipgeneid = pd.merge(cut090uniprot, unip2geneid, how='inner', left_on=['uniprot'], right_on = ['UniProtKB'])
print(cut090unipgeneid.shape)
print(cut090unipgeneid.keys())
print(cut090unipgeneid.head())

geneid2diseasetmp = pd.read_csv('all_gene_disease_associations.tsv', sep="\t", header=0, index_col=False)

geneid2disease = geneid2diseasetmp[['geneId','diseaseName','diseaseSemanticType']]

print(geneid2disease.shape)
print(geneid2disease.keys())
print(geneid2disease.head())

cut090diseasetmp = pd.merge(cut090unipgeneid, geneid2disease, how='inner', left_on=['GENEID'], right_on = ['geneId'])

cut090disease = cut090diseasetmp[['uniprot','GENEID','diseaseName','diseaseSemanticType']]

print(cut090disease.shape)
print(cut090disease.keys())
print(cut090disease.head())


cut090disease.to_csv('cut090disease.csv')

rankdisease = cut090disease[['GENEID','diseaseName']].groupby('diseaseName').agg('count').sort_values('GENEID',ascending=False)

rankdisease.reset_index()
rankdisease.to_csv('rankeddisease.csv')

ranksemantic = cut090disease[['GENEID','diseaseSemanticType']].groupby('diseaseSemanticType').agg('count').sort_values('GENEID',ascending=False)

ranksemantic.reset_index()
ranksemantic.to_csv('rankedsemantic.csv')

