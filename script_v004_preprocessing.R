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

#------------------------- Processing DATASET 
# Cancer types 
diseaseAbbrvs <- c("ACC", "BLCA", "CHOL", "ESCA",  "HNSC","LGG", "LIHC", "LUSC", "MESO", "PAAD",  "PRAD",  "SARC", "SKCM",  "TGCT", "UCS")
diseaseAbbrvs_lower <- c("acc", "chol", "blca", "esca", "hnsc", "lgg", "lihc", "lusc", "meso", "paad", "prad", "sarc", "skcm",  "tgct", "ucs")
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

#clinical dataset with the filter for metastase
bd.d = subset(bd.c, new_tumor_event_dx_indicator == "YES" | new_tumor_event_dx_indicator == "NO")
table(as.character(bd.d$abr), as.character(bd.d$new_tumor_event_dx_indicator))
tab = data.frame(table(as.character(bd.d$abr)))


#------------ RNA

#creating function to extract the raw_count 
load_rna <- function(fname2){
  bd = read.csv(fname2, sep='\t',header = F)
  remove.c = c()
  patients = c()
  for(i in 2:dim(bd)[2]){
    if(bd[2,i]!='raw_count'){
      remove.c = c(remove.c,i)
    }else{
      patients = c(patients,as.character(bd[1,i]))
    }
  }
  bd.aux = data.frame(bd[,-remove.c])
  #bd.aux = bd.aux[,]
  dim(bd.aux)
  bd.aux = bd.aux[-c(1,2),]
  bd.aux$V1 = as.character(bd.aux$V1)
  bd.aux$V1 = gsub("?|","g",bd.aux$V1, fixed = T)
  colname = bd.aux$V1
  bd.aux = data.frame(t(bd.aux[,-1]))
  names(bd.aux) = colname
  head(bd.aux[,c(1:10)])
  bd.aux = data.frame(patients, bd.aux)
  return (bd.aux)
}

#Rotine to extract all RNA counts from the cancer types selected 
bd.e = load_rna(fname2[1])
tab[tab$Var1==diseaseAbbrvs[1],]
for(i in 2:length(fname2)){
  print(diseaseAbbrvs[i])
  dim(bd.e)
  bd.e = rbind(bd.e,load_rna(fname2[i]))
  tab[tab$Var1==diseaseAbbrvs[i],]
  dim(bd.e)
}

#------------------------- Savinf files

#write.table(bd.d, "clinical.txt", row.names = F, sep=';')
write.table(bd.e, "Data\\rnaseq.txt", sep=';',row.names = F)

#------------------------- Extra filters 
#keeping only the patients present in both datasets
#Temp local
temp = "C:\\Users\\raque\\Google Drive\\SFU\\Project 2 - Spring 2019\\Data"

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
#write.table(bd.rna3, paste(temp,'\\tcga_cli.txt',sep=''),sep=';')
write.table(bd.rna4, paste(temp,'\\tcga_rna.txt',sep=''),sep=';')



