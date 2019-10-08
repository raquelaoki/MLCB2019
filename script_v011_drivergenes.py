'''Loading libraries'''
import pandas as pd 
import numpy as np 
import os

'''
Notes:
1) some genes have classificatio as known and predicted driver. 
The classification probably changes depending on the cancer type. 
2) should I have one outcome model for cancer type and one model for all 
    and compare the results? 

Tasks: 
a) Subset cancer types I'm interest on, removing duplicateds 
b) the evaluation depends on cancer type x gene, not only gene

'''



#All cancer types
remove = ["known_match","chr","strand_orig",'info', 'region', 'strand', 'transcript', 'cdna','pos','ref', 
          'alt', 'default_id', 'input','transcript_aminoacids', 'transcript_exons','protein_change',
          'exac_af', 'Pfam_domain', 'cadd_phred',"gdna","protein","consequence","protein_pos",
          "is_in_delicate_domain","inframe_driver_mut_prediction","is_in_cluster","missense_driver_mut_prediction",
          "driver_mut_prediction","mutation_location","disrupting_driver_mut_prediction",'exon',
          "alt_type","known_oncogenic","known_predisposing",'sample']


path = 'C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\driver_genes\\'
files = []

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.tsv' in file:
            files.append(os.path.join(r, file))

count = 0
for f in files: 
    dgenes0 = pd.read_csv(f, sep='\t')
    dgenes0 = dgenes0.drop(remove, axis = 1)
    if count==0:
        dgenes = dgenes0
        count = 1
    else: 
        dgenes = dgenes.append(pd.DataFrame(data = dgenes0), ignore_index=True)



d_statment = ['predicted passenger','not protein-affecting','predicted driver: tier 1','predicted driver: tier 2','polymorphism']
dgenes.driver_statement[~dgenes.driver_statement.isin(d_statment)]='known'
dgenes.driver_statement.value_counts()
dgenes.cancer.value_counts()

print('testing group by\n')
print(dgenes.groupby(["driver", "driver_statement"]).size().reset_index(name="Time"))
print(dgenes.groupby(["driver", "driver_gene"]).size().reset_index(name="Time"))

dgenes = dgenes.sort_values(by=['gene'])
dgenes = dgenes.dropna(subset=['driver_gene','driver','gene_role'])

print('removing duplicateds\n')
print(dgenes.shape)
print(dgenes.head())
dgenes = dgenes.drop(['driver_statement','cancer'],axis=1)
dgenes.drop_duplicates(keep = 'first', inplace = True)  
print(dgenes.shape)
print(dgenes.head())





