rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#AUTHOR: Raquel Aoki
#DATE: 2019/03/07
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#Workdiretory 
setwd("C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019")

#------------------------- DOWNLOAD DATASET 
# check reference on email sent to Olga on March 2019
# Almost no modifications in the download section of this code

theRootDir <- "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data"
diseaseAbbrvs <- c("ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "COADREAD", "DLBC", "ESCA", "FPPP", "GBM", 
                   "GBMLGG", "HNSC", "KICH", "KIPAN", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", 
                   "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD", "STES", "TGCT", "THCA", 
                   "THYM", "UCEC", "UCS", "UVM")


#------------------------ RNA-SEQ / GENE EXPRESSION 
missingAbrvsRnaSeq <- c(10, 31) # there is no RNA-seq data for "FPPP" or "STAD"
rnaSeqDiseaseAbbrvs <- diseaseAbbrvs[-missingAbrvsRnaSeq]
rnaSeqFilesDir <- paste(theRootDir, "rnaSeq/", sep="")
dir.create(rnaSeqFilesDir, showWarnings = FALSE) # make this directory if it doesn't exist.
for(i in 1:length(rnaSeqDiseaseAbbrvs))
{
  fname <- paste("gdac.broadinstitute.org_", rnaSeqDiseaseAbbrvs[i], ".Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2015082100.0.0.tar.gz", sep="")
  download.file(paste("http://gdac.broadinstitute.org/runs/stddata__2015_08_21/data/", rnaSeqDiseaseAbbrvs[i], "/20150821/gdac.broadinstitute.org_", rnaSeqDiseaseAbbrvs[i], ".Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2015082100.0.0.tar.gz", sep=""), paste(rnaSeqFilesDir, fname, sep=""))
}

# Unzip the downloaded ".tar.gz" RNA-seq data! NB, this command has been tested in Linux. It may not work in Windows. If it does not work, please extract these files manually using software such as 7zip.
thegzFiles <-  paste(rnaSeqFilesDir, dir(rnaSeqFilesDir), sep="")
sapply(thegzFiles, untar, exdir=rnaSeqFilesDir)


#------------------------ CLINICAL INFORMATION 

clinicalFilesDir <- paste(theRootDir, "clinical/", sep="")
dir.create(clinicalFilesDir, showWarnings = FALSE) # make this directory if it doesn't exist.

allTcgaClinAbrvs <- c("acc", "blca", "brca", "cesc", "chol", "cntl", "coad", "dlbc", "esca", "fppp", "gbm", "hnsc", "kich", 
                      "kirc", "kirp", "laml", "lcml", "lgg", "lihc", "lnnh", "luad", "lusc", "meso", "misc", "ov", "paad", 
                      "pcpg", "prad", "read", "sarc", "skcm", "stad", "tgct", "thca", "thym", "ucec", "ucs", "uvm")
for(i in 1:length(allTcgaClinAbrvs))
{
  fname <- paste("nationwidechildrens.org_clinical_patient_", allTcgaClinAbrvs[i], ".txt", sep="")
  theUrl <- paste("https://raw.github.com/paulgeeleher/tcgaData/master/nationwidechildrens.org_clinical_patient_", allTcgaClinAbrvs[i], ".txt", sep="")
  download.file(theUrl, paste(clinicalFilesDir, fname, sep=""))
}


#NOTE: columns have different names in each cancer type. These are more commom among them all
allTcgaClinAbrvs <- c("acc", "chol",'BRCA' ,"blca", "esca", "hnsc", "lgg", "lihc", "lusc", "meso", "paad", "prad", "sarc", "skcm",  "tgct", "ucs")

fname <- paste("nationwidechildrens.org_clinical_patient_",allTcgaClinAbrvs,".txt" , sep='')
cnames = c("bcr_patient_barcode",
           "gender", 
           "race" , 
           "ethnicity", 
           "tumor_status", 
           "vital_status",  
           'new_tumor_event_dx_indicator') 

print(paste("columns:",length(cnames)))
for(i in 1:length(fname)){
  bd = read.csv(fname[i], sep = "\t") 
  print(paste(i, '-', length(intersect(names(bd),cnames))))
  print(setdiff(cnames,names(bd)))
}

#i = i + 1
#bd = read.csv(fname[i], sep = "\t") 
#names(bd) 


#------------------------ DOWNLOAD SOMATIC MUTATION 
diseaseAbbrvsForMuts <- c("ACC", "BLCA", "BRCA", "CHOL", "ESCA", "HNSC", "LGG", "LIHC", "LUSC", "PAAD", "PRAD", "SARC", "SKCM", "TGCT", "UCS")
mutFilesDir <- paste(theRootDir, "\\mutation_data", sep="")
dir.create(mutFilesDir, showWarnings = FALSE) # make this directory if it doesn't exist.
for(i in 1:length(diseaseAbbrvsForMuts))
{
  mutationDataUrl <- paste("http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/", diseaseAbbrvsForMuts[i], "/20160128/gdac.broadinstitute.org_", diseaseAbbrvsForMuts[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz", sep="")
  fname <- paste("gdac.broadinstitute.org_", diseaseAbbrvsForMuts[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz", sep="")
  download.file(mutationDataUrl, paste(mutFilesDir, fname, sep=""))
}
thegzFiles <-  paste(mutFilesDir, dir(mutFilesDir), sep="\\")
sapply(thegzFiles, untar, exdir=mutFilesDir)



#Test 
library(TCGAbiolinks)






