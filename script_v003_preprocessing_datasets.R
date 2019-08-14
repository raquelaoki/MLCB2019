rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#AUTHOR: Raquel Aoki
#DATE: 2019/03/07
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#------------------------- Work diretory 
#setwd("C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019")
setwd("~/GitHub/project_spring2019")

#------------------------- References
# check reference on email sent to Olga on March 2019
# Almost no modifications in the download section of this code
#paulgeeleher, firebrownse 


#because the model I'm using it's necessary work with the RAW COUNT. 
#gene expression is usually modeled as a negative binomial, a discrete distribution. 

#------------------------- Processing DATASET 
# Cancer types 
diseaseAbbrvs <- c("ACC", "BLCA",'BRCA', "CHOL", "ESCA",  "HNSC","LGG", "LIHC", "LUSC", "MESO", "PAAD",  "PRAD",  "SARC", "SKCM",  "TGCT", "UCS")
diseaseAbbrvs_lower <- c("acc", "chol",'brca' ,"blca", "esca", "hnsc", "lgg", "lihc", "lusc", "meso", "paad", "prad", "sarc", "skcm",  "tgct", "ucs")
#there are more cancer types available, I filtered these because I think they are more balanced between metastase/not metastase 


#Files names
fname <- paste("Data\\clinical\\nationwidechildrens.org_clinical_patient_",diseaseAbbrvs_lower,".txt" , sep='')
fname2 <- paste("Data\\rnaSeq\\gdac.broadinstitute.org_",diseaseAbbrvs, "\\" , diseaseAbbrvs,'.rnaseqv2.txt',sep='')
#For the RNAseq I had to manually change a bit the folder and files names. It was very long and R weren't reading it.


#------------ Clinical Information

#Clinical information names 
cnames = c("bcr_patient_barcode","gender", 'race' , "ethnicity",  "vital_status", "tumor_status",'new_tumor_event_dx_indicator','abr')  
#metastases is new_tumor_event
#check each cancer type for metastases, clinical_stage, dasys to birth and days to death

#Rotine to read the files, select the important features, and bind in a unique dataset
i = 1
bd.aux = read.csv(fname[i], sep = "\t") 
bd.aux = subset(bd.aux, gender== "MALE" | gender == 'FEMALE')
bd.aux$abr = diseaseAbbrvs[i]
cnames1 = c(cnames)
bd.c = subset(bd.aux, select = cnames)

for(i in 2:length(fname)){
  bd.aux = read.csv(fname[i], sep = "\t", header = T) 
  bd.aux = subset(bd.aux, gender== "MALE" | gender == 'FEMALE')
  bd.aux$abr = diseaseAbbrvs[i]
  #(paste(i, '-', length(intersect(names(bd.aux),cnames))))
  #print(setdiff(cnames,names(bd.aux)))
  bd.c = rbind(bd.c, 
               subset(bd.aux, select = cnames))
}

#clinical dataset 
head(bd.c)


#------ EXPLORATION ONLY, NOT IMPORTANT

#Compare with new clinical information 
abr_newcl <- c("acc", "chol",'brca' ,"blca", "esca", "hnsc", "lihc", "lusc", "meso", "paad", "prad", "sarc", "skcm",  "tgct")


columns = c('Patient.ID','Cancer.Type','Sex','Neoplasm.American.Joint.Committee.on.Cancer.Clinical.Distant.Metastasis.M.Stage','American.Joint.Committee.on.Cancer.Metastasis.Stage.Code',
            'Neoplasm.American.Joint.Committee.on.Cancer.Clinical.Distant.Metastasis.M.Stage','Metastatic.disease.confirmed')
i = 1
clinical = read.table(paste('Data\\clinical\\',abr_newcl[i],'_tcga_clinical_data.tsv',sep=''),sep='\t' ,header = T)
col = intersect(columns, names(clinical))
if(length(col)==4){
  clinical = subset(clinical, select = col)
}else{
  cat(paste('ERROR ON ',abr_newcl[i], ' - ',i,sep=''))
}
names(clinical)[4] = 'Metastasis_M'
clinical$Cancer.Study  = paste(abr_newcl[i],'_tcga',sep='')

for(i in 2:length(abr_newcl)){
  clinical1 = read.table(paste('Data\\clinical\\',abr_newcl[i],'_tcga_clinical_data.tsv',sep=''),sep='\t' ,header = T)
  col = intersect(columns, names(clinical1))
  if(i==6 | i == 14){
    col = col[1:4]
  }
  if(length(col)==4){
    clinical1 = subset(clinical1, select = col)
  }else{
    cat(paste('ERROR ON ',abr_newcl[i], ' - ',i,sep=''))
  }
  names(clinical1)[4] = 'Metastasis_M'
  clinical1$Cancer.Study  = paste(abr_newcl[i],'_tcga',sep='')
  clinical = rbind(clinical,clinical1)
}

table(clinical$Metastasis_M)
clinical$Metastasis_M = as.character(clinical$Metastasis_M)
clinical$Metastasis_M[clinical$Metastasis_M=='M1a'|clinical$Metastasis_M=='M1b'|clinical$Metastasis_M=='M1c'] = 'M1'
clinical$Metastasis_M[clinical$Metastasis_M=='YES'] = 'M1'
clinical$Metastasis_M[clinical$Metastasis_M=='NO'] = 'M0'
clinical$Metastasis_M[clinical$Metastasis_M=='cM0 (i+)'] = 'M0'
clinical$Metastasis_M[clinical$Metastasis_M==''] = 'MX'
table(clinical$Metastasis_M)


##Legend: 
#M0 means that no distant cancer spread was found.
#M1 means that the cancer has spread to distant organs or tissues (distant metastases were found).
#MX for unknown status of metastatic disease. 
#cM0: any case without clinical or pathologic evidence of metastases is to be classified as clinically M0

#Merging old and new clinical information 
clinical$Patient.ID = as.character(clinical$Patient.ID)
bd.c1 = merge(bd.c,clinical, by.y = 'Patient.ID' ,by.x ='bcr_patient_barcode',all=T)
dim(bd.c); dim(clinical); dim(bd.c1)
bd.c1$new_tumor_event_dx_indicator = as.character(bd.c1$new_tumor_event_dx_indicator)

table(bd.c1$new_tumor_event_dx_indicator,bd.c1$Metastasis_M)

#clinical dataset with the filter for metastase
bd.d = subset(bd.c, new_tumor_event_dx_indicator == "YES" | new_tumor_event_dx_indicator == "NO")
table(as.character(bd.d$abr), as.character(bd.d$new_tumor_event_dx_indicator))
tab = data.frame(table(as.character(bd.d$abr)))

clinical$Patient.ID = as.character(clinical$Patient.ID)
bd.d$bcr_patient_barcode = as.character(bd.d$bcr_patient_barcode)
test = merge(bd.d, clinical, by.y = 'Patient.ID',by.x ='bcr_patient_barcode',all.y = T)
test$new_tumor_event_dx_indicator[is.na(test$new_tumor_event_dx_indicator)]='[Unknown]'

test$new_tumor_event_dx_indicator = as.character(test$new_tumor_event_dx_indicator)
table(test$new_tumor_event_dx_indicator,test$Metastasis_M)*100/dim(test)[1]
head(subset(test,is.na(gender)))
dim(subset(test,is.na(Sex)))

#------ EXPLORATION ONLY, NOT IMPORTANT
#Conclusion: use the new_tumor_event_dx_indicator

bd.c$new_tumor_event_dx_indicator  = as.character(bd.c$new_tumor_event_dx_indicator)
bd.c1 = subset(bd.c, new_tumor_event_dx_indicator=='YES' | new_tumor_event_dx_indicator == 'NO')
bd.c1 = subset(bd.c1 , select = c( bcr_patient_barcode, gender,  new_tumor_event_dx_indicator, abr))

write.table(bd.c1,'C://Users//raoki//Documents//GitHub//project_spring2019//Data//tcga_cli1.txt', row.names = F, sep = ';')


#------------ RNA

#creating function to extract the scaled_estimate 
load_rna <- function(fname2){
  bd = read.csv(fname2, sep='\t',header = F)
  remove.c = c()
  patients = c()
  for(i in 2:dim(bd)[2]){
    if(as.character(bd[2,i])!='raw_count'){#scaled_estimate
      remove.c = c(remove.c,i)
    }else{
      patients = c(patients,as.character(bd[1,i]))
    }
  }
  
  #removing other columns
  bd.aux = data.frame(bd[,-remove.c])
  #removing first lines 
  bd.aux = bd.aux[-c(1,2),]
  #organizing the columns names 
  colname = strsplit(as.character(bd.aux$V1), split = "|", fixed = T)
  gene = c()
  gene_id = c()
  for(i in 1:length(colname)){
    gene[i] = colname[[i]][1]
    gene_id[i] = colname[[i]][2]
  }
  gene[gene=='?'] = paste('g',gene_id[gene=='?'],sep='')
  #transpose dataset
  bd.aux = data.frame(t(bd.aux[,-1]))
  names(bd.aux) = gene
  row.names(bd.aux)=NULL
  for(i in 1:dim(bd.aux)[2]){
    bd.aux[,i]= format(as.numeric(as.character(bd.aux[,i])), digits = 17)
  }
  bd.aux = data.frame(patients, bd.aux)
  return (bd.aux)
}


#Rotine to extract all RNA counts from the cancer types selected 
bd.e  = load_rna(fname2[1])
print(paste('total ->', length(fname2)))
tab[tab$Var1==diseaseAbbrvs[1],]
for(i in 2:length(fname2)){
  print(paste(i,'-' ,diseaseAbbrvs[i]))
  bd.aux = load_rna(fname2[i])
  #checking if the columns names are the same and if they are in the same order
  print(paste('columns: ', dim(bd.e)[2], "---", sum(names(bd.aux)==names(bd.e))))
  if(sum(names(bd.aux)==names(bd.e))!=dim(bd.e)[2]){
    print('Error!')
  }
  bd.e = rbind(bd.e,bd.aux)
  write.csv(bd.e,"Data\\rnaseq_intermediate.txt", sep=';', row.names = F)  
}

#------------------------- Saving files

write.table(bd.d, "Data\\clinical.txt", row.names = F, sep=';')
write.table(bd.e, "Data\\rnaseq.txt", sep=';',row.names = F)

#------------------------- Extra filters 
#keeping only the patients present in both datasets
#Temp local
#temp = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data"
temp = "Data"

#reading datasets 
bd.rna = read.csv(paste(temp, "\\rnaseq.txt",sep=""),sep=';')
bd.cli = read.csv(paste(temp,"\\clinical.txt",sep=""),sep=';')

#creating a auxliar databade
#bd.aux = bd.cli[,c(1,2)]
names(bd.cli)[1]='patients'
bd.cli$aux = 1

#fixind the patient id
bd.rna$patients =  as.character(bd.rna$patients)
bd.rna$patients2 =  as.character(bd.rna$patients)

bd.rna$patients = gsub('-01A-.*','',bd.rna$patients)
bd.rna$patients = gsub('-01B-.*','',bd.rna$patients)
bd.rna$patients = gsub('-02A-.*','',bd.rna$patients)
bd.rna$patients = gsub('-02B-.*','',bd.rna$patients)
bd.rna$patients = gsub('-05A-.*','',bd.rna$patients)
bd.rna$patients = gsub('-06A-.*','',bd.rna$patients)
bd.rna$patients = gsub('-06B-.*','',bd.rna$patients)
bd.rna$patients = gsub('-07A-.*','',bd.rna$patients)
bd.rna$patients = gsub('-11A-.*','',bd.rna$patients)
bd.rna$patients = gsub('-11B-.*','',bd.rna$patients)

#removing duplicateds 
bd.rna = bd.rna[order(bd.rna$patients2),]
bd.rna = subset(bd.rna, !duplicated(patients))


bd.rna2 = merge(bd.rna, bd.cli, by = 'patients', all = T)
bd.rna2 = subset(bd.rna2,!is.na(aux) & !is.na(patients2))
#write.table(bd.rna2, paste(temp,'\\tcga_clinical_rna.txt',sep=''),sep=';')

#Saving in 2 separated files 
bd.rna3 = subset(bd.rna2, select = names(bd.cli))
bd.rna4 = subset(bd.rna2, select = names(bd.rna))

dim(bd.rna3)
dim(bd.rna4)
head(bd.rna4[,c(1:10)])
write.table(bd.rna3, paste(temp,'\\tcga_cli.txt',sep=''),sep=';')
write.table(bd.rna4, paste(temp,'\\tcga_rna.txt',sep=''),sep=';')


#### Addinting log transformation
bd = read.table('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\data_final.csv',header=T,sep=',')

for(i in 20:dim(bd)[2]){
  bd[,i] = log(bd[,i]+1)
}
write.table(bd,'C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\data_final_log.csv',row.names = F, sep=',')




# #------------------------ PROCESING MAF FILES / MUTATION
# Reference
#INSTALLING PACKAGES
if (!require("BiocManager"))
  install.packages("BiocManager")
BiocManager::install("maftools")
library(maftools)

#LOADING CLINICAL INFORMATION 
clinical = read.csv('file:///C:/Users/raoki/Documents/GitHub/project_spring2019/Data/tcga_cli1.txt', sep = ';') #incomplete
names(clinical)[1] = 'Tumor_Sample_Barcode'


diseaseAbbrvsForMuts <- c("ACC", "BLCA", "BRCA", "CHOL", "ESCA", "HNSC", "LGG", "LIHC", "LUSC", "PAAD", "PRAD", "SARC", "SKCM", "TGCT", "UCS")
mutFilesDir <- paste('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data', "\\mutation_data", sep="")
exception = c("TCGA-P5-A5F6","TCGA-EJ-A7NG","TCGA-NA-A4QY")

for(i in 1:length(diseaseAbbrvsForMuts))
{
  mutationDataUrl <- paste(mutFilesDir,"\\gdac.broadinstitute.org_", diseaseAbbrvsForMuts[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0", sep="")
  setwd(mutationDataUrl)
  manifest = read.table('MANIFEST.txt')
  for(j in 1:dim(manifest)[1]){
    barcode = substr(manifest$V2[j],0,12)
    cinfo = subset(clinical, Tumor_Sample_Barcode==barcode)
    if(sum(barcode==exception)==0){
      if(dim(cinfo)[1]==1){
        mutation = read.maf(maf = as.character(manifest$V2[j]), clinicalData = cinfo)
      }else{
        mutation = read.maf(maf = as.character(manifest$V2[j]))
      }
      if(j==1 & i==1){
        genes = getGeneSummary(mutation)
        genes = subset(genes, select = c('Hugo_Symbol','total'))
        names(genes)[2] = barcode
      }else{
        genes0 = getGeneSummary(mutation)
        genes0 = subset(genes0, select = c('Hugo_Symbol','total'))
        names(genes0)[2] = barcode
        genes = merge(genes,genes0,by = 'Hugo_Symbol',all=T)
      }
    }
  }
  
}

write.table(genes, file = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\tcga_mu.txt",row.names = F, sep=',')

count = 0 
for(i in 1:length(diseaseAbbrvsForMuts))
{
  mutationDataUrl <- paste(mutFilesDir,"\\gdac.broadinstitute.org_", diseaseAbbrvsForMuts[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0", sep="")
  setwd(mutationDataUrl)
  manifest = read.table('MANIFEST.txt')
  count = count + dim(manifest)[1]
  
}

#Remove NA for 0 
genes[is.na(genes)] <- 0
genes = genes[genes$Hugo_Symbol!='.']


write.table(genes, file = "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\tcga_mu.txt",row.names = F, sep=',')


#MERGE CLINICAL INFORMATION AND MUTATION
bd = read.csv('C://Users//raoki//Documents//GitHub//project_spring2019//Data//tcga_mu.txt', header=T, sep=',')
cl = read.csv('C://Users//raoki//Documents//GitHub//project_spring2019//Data//tcga_cli1.txt', header = T, sep=';')

#Transposing mutation dataset and fixing patient
bd1 = t(bd)
bd1 = data.frame(rownames(bd1),bd1)
rownames(bd1) = NULL
for( i in 1:dim(bd1)[2]){
  names(bd1)[i] = as.character(bd1[1,i])
}
bd1 = bd1[-1,]
bd1$Hugo_Symbol = as.character(bd1$Hugo_Symbol)
bd1$Hugo_Symbol = gsub(pattern = '.', replacement = '-',bd1$Hugo_Symbol, fixed = T)

head(bd1[,c(1:10)])


#Creating a variable indicator with the prediction value in the 0/1 format
cl$y = as.character(cl$new_tumor_event_dx_indicator)
cl$y[cl$y=="NO"] = 0 
cl$y[cl$y=="YES"] = 1 

cl$gender = as.character(cl$gender)
cl$gender[cl$gender=='MALE'] = 0 
cl$gender[cl$gender=='FEMALE'] = 1

#selecting important clinical features and creating dummy/binary variables for future use
cl_s = subset(cl, select = c('bcr_patient_barcode','gender','abr','y'))

# MERGE
bd2 = merge(bd1, cl_s, by.x ='Hugo_Symbol',by.y = 'bcr_patient_barcode' , all=T)
# CHECK DIFFERENT SIZE OF DATASET (PROBLABLY NEED TO RUN CL AGAIN WITH BLRC)

a = subset(bd2, is.na(gender))$Hugo_Symbol
b = subset(bd2, is.na(A1BG))$Hugo_Symbol
a = a[order(a)]
b = b[order(b)]

test1 = subset(bd2, !is.na(gender) & !is.na(A1BG))
test2 = subset(bd2, !is.na(gender) & is.na(A1BG))
test3 = subset(bd2, is.na(gender)  & !is.na(A1BG))

hnsc = subset(test2, abr=='HNSC'))$Hugo_Symbol



b1 = as.character(test2$HNSC.rnaseqv2) #mutation
b2 = as.character(test3$HNSC.rnaseqv2) #clinical info
b3a= c()
b3 = read.table('file:///C:/Users/raoki/Documents/GitHub/project_spring2019/Data/rnaSeq/gdac.broadinstitute.org_HNSC/HNSC.rnaseqv2.txt', sep='\t',header = F)
for(i in 2:dim(b3)[2]){
  b3a = c(b3a, as.character(b3[1,i]))
}
  
