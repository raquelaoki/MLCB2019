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

theRootDir <- "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data"
diseaseAbbrvs <- c("ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "COADREAD", "DLBC", "ESCA", "FPPP", "GBM", 
                   "GBMLGG", "HNSC", "KICH", "KIPAN", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", 
                   "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD", "STES", "TGCT", "THCA", 
                   "THYM", "UCEC", "UCS", "UVM")

#eliminited some weid download datasets
allTcgaClinAbrvs <- c("acc", "blca", "brca", "cesc", "chol", "coad", "dlbc", "esca",  
                      "gbm", "hnsc", "kich", "kirc", "kirp", "lgg", "lihc", "luad", "lusc", 
                      "meso", "ov", "paad", "pcpg", "prad", "read", "sarc", "skcm", "stad", "tgct", 
                      "thca", "thym", "ucec", "ucs", "uvm") 

#no tumor status , "laml"


fname <- paste("nationwidechildrens.org_clinical_patient_",allTcgaClinAbrvs,".txt" , sep='')


#Selecting important features
#There are 33 cancers, might be easier to do manually one by one
#After standartize, than we can see which cancers make sense
#than work with RnA 


fname <- paste("Data\\clinical\\nationwidechildrens.org_clinical_patient_",allTcgaClinAbrvs,".txt" , sep='')
cnames = c("bcr_patient_barcode","gender", 'race' , "ethnicity",  "vital_status", "tumor_status",'new_tumor_event_dx_indicator','abr')  
#metastases is new_tumor_event

#check each cancer type for metastases, clinical_stage, dasys to birth and days to death

i = 1
bd.aux = read.csv(fname[i], sep = "\t") 
bd.aux = subset(bd.aux, gender== "MALE" | gender == 'FEMALE')
bd.aux$abr = allTcgaClinAbrvs[i]
cnames1 = c(cnames)
bd.c = subset(bd.aux, select = cnames)


for(i in 2:length(fname)){
  bd.aux = read.csv(fname[i], sep = "\t", header = T) 
  bd.aux = subset(bd.aux, gender== "MALE" | gender == 'FEMALE')
  bd.aux$abr = allTcgaClinAbrvs[i]
  #(paste(i, '-', length(intersect(names(bd.aux),cnames))))
  #print(setdiff(cnames,names(bd.aux)))
  bd.c = rbind(bd.c, 
               subset(bd.aux, select = cnames))
}

i = i + 1
bd.aux = read.csv(fname[i], sep = "\t") 
names(bd.aux) 


for(i in 2:length(fname)){
  bd.aux = read.csv(fname[i], sep = "\t", header = T) 
  bd.aux = subset(bd.aux, gender== "MALE" | gender == 'FEMALE')
  bd.aux$abr = allTcgaClinAbrvs[i]
  #(paste(i, '-', length(intersect(names(bd.aux),cnames))))
  print(setdiff(cnames1,names(bd.aux)))
 # bd.c = rbind(bd.c, 
  #             subset(bd.aux, select = cnames))
}

