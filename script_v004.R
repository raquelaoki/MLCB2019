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
                      "gbm", "hnsc", "kich", "kirc", "kirp", "laml", "lgg", "lihc", "luad", "lusc", 
                      "meso", "ov", "paad", "pcpg", "prad", "read", "sarc", "skcm", "stad", "tgct", 
                      "thca", "thym", "ucec", "ucs", "uvm") 


fname <- paste("nationwidechildrens.org_clinical_patient_",allTcgaClinAbrvs,".txt" , sep='')


#Selecting important features
#There are 33 cancers, might be easier to do manually one by one
#After standartize, than we can see which cancers make sense
#than work with RnA 


fname <- paste("Data\\clinical\\nationwidechildrens.org_clinical_patient_",allTcgaClinAbrvs,".txt" , sep='')
cnames = c("bcr_patient_barcode",
           "gender", 
           "race" , 
           "ethnicity", 
           "tumor_status", 
           "vital_status",  
          # "metastatic_dx_confirmed_by" ,  
          # "metastatic_dx_confirmed_by_other", 
           "metastatic_tumor_site" , #metastasis_site
           "clinical_stage" ,  
           "days_to_birth" ,#birth_days_to
           "days_to_death" ) #death_days_to


cnames1 = c("bcr_patient_barcode",
           "gender", 
           "race" , 
           "ethnicity", 
           "tumor_status", 
           "vital_status",  
           "metastasis_site" , 
           "clinical_stage" ,  
           "birth_days_to" ,
           "death_days_to" ) 

print(paste("columns:",length(cnames)))
for(i in 1:length(fname)){
  bd = read.csv(fname[i], sep = "\t") 
  print(paste(i, '-', length(intersect(names(bd),cnames))))
  print(setdiff(cnames,names(bd)))
}

i = i + 1
bd = read.csv(fname[i], sep = "\t") 
print(setdiff(cnames1,names(bd)))
names(bd) 



