rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#AUTHOR: Raquel Aoki
#DATE: 2019/03/07
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#Workdiretory 
setwd("C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019")

#------------------------- Processing DATASET 
# check reference on email sent to Olga on March 2019
# Almost no modifications in the download section of this code

#theRootDir <- "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data"
#diseaseAbbrvs <- c("ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD",  "DLBC", "ESCA",
#                   "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LGG", "LIHC", "LUAD", "LUSC", 
#                   "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD", "TGCT", 
#                   "THCA",  "THYM", "UCEC", "UCS", "UVM")

#eliminited some weid download datasets
#allTcgaClinAbrvs <- c("acc", "blca", "brca", "cesc", "chol", "coad", "dlbc", "esca",  
#                      "gbm", "hnsc", "kich", "kirc", "kirp", "lgg", "lihc", "luad", "lusc", 
#                      "meso", "ov", "paad", "pcpg", "prad", "read", "sarc", "skcm", "stad", "tgct", 
#                      "thca", "thym", "ucec", "ucs", "uvm") 

diseaseAbbrvs <- c("ACC", "BLCA", "CHOL", "ESCA",  "HNSC","LGG", "LIHC", "LUSC", "MESO", "PAAD",  "PRAD",  "SARC", "SKCM",  "TGCT", "UCS")

diseaseAbbrvs_lower <- c("acc", "chol", "blca", "esca", "hnsc", "lgg", "lihc", "lusc", "meso", "paad", "prad", "sarc", "skcm",  "tgct", "ucs")

fname <- paste("nationwidechildrens.org_clinical_patient_",diseaseAbbrvs_lower,".txt" , sep='')


#Selecting important features
#There are 33 cancers, might be easier to do manually one by one
#After standartize, than we can see which cancers make sense
#than work with RnA 


fname <- paste("Data\\clinical\\nationwidechildrens.org_clinical_patient_",diseaseAbbrvs_lower,".txt" , sep='')
cnames = c("bcr_patient_barcode","gender", 'race' , "ethnicity",  "vital_status", "tumor_status",'new_tumor_event_dx_indicator','abr')  
#metastases is new_tumor_event

#check each cancer type for metastases, clinical_stage, dasys to birth and days to death

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


head(bd.c)



#test

bd.d = subset(bd.c, new_tumor_event_dx_indicator == "YES" | new_tumor_event_dx_indicator == "NO")
table(as.character(bd.d$abr), as.character(bd.d$new_tumor_event_dx_indicator))


#-------RNA
#__data.data
fname2 <- paste("Data\\rnaSeq\\gdac.broadinstitute.org_",diseaseAbbrvs,
                "\\" , diseaseAbbrvs,'.rnaseqv2.txt',sep='')


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


bd.e = load_rna(fname2[1])
for(i in 2:length(fname2)){
  bd.e = rbind(bd.e,load_rna(fname2[1]))
}

#missing genes names
