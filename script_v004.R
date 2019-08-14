rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#AUTHOR: Raquel Aoki
#DATE: 2019/08
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#Workdiretory 
setwd("C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019")


#INSTALLING PACKAGES
if (!require("BiocManager"))
  install.packages("BiocManager")
BiocManager::install("maftools")
library(maftools)

#------------------------- DOWNLOAD DATASET 
# check reference on email sent to Olga on March 2019
# Almost no modifications in the download section of this code

theRootDir <- "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew"

# Cancer types 
diseaseAbbrvs <- c("ACC", "BLCA",'BRCA', "CHOL", "ESCA",  "HNSC","LGG", "LIHC", "LUSC", "MESO", "PAAD",  "PRAD",  "SARC", "SKCM",  "TGCT", "UCS")
diseaseAbbrvs_lower <- c("acc", "chol",'brca' ,"blca", "esca", "hnsc", "lgg", "lihc", "lusc", "meso", "paad", "prad", "sarc", "skcm",  "tgct", "ucs")

#------------------------  SOMATIC MUTATION - donwload only once
#for(i in 1:length(diseaseAbbrvs)){
#  mutationDataUrl <- paste("http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/", diseaseAbbrvs[i], "/20160128/gdac.broadinstitute.org_", diseaseAbbrvs[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz", sep="")
#  fname <- paste("gdac.broadinstitute.org_", diseaseAbbrvs[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz", sep="")
#  download.file(mutationDataUrl, paste(theRootDir, fname, sep="\\"))
#}


#------------------------ CLINICAL INFORMATION 
cnames = c("bcr_patient_barcode",
#           "gender", 
#           "ethnicity", 
#           "tumor_status", 
#           "vital_status",  
           'new_tumor_event_dx_indicator') 

for(i in 1:length(diseaseAbbrvs)){
  fname <- paste("nationwidechildrens.org_clinical_patient_", diseaseAbbrvs[i], ".txt", sep="")
  theUrl <- paste("https://raw.github.com/paulgeeleher/tcgaData/master/nationwidechildrens.org_clinical_patient_", diseaseAbbrvs_lower[i], ".txt", sep="")
  cli = read.csv(theUrl, sep = '\t', header = T)[-c(1,2),]
  cli = subset(cli, select = intersect(names(cli),cnames))
}









