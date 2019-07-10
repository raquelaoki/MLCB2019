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
# for(i in 1:length(allTcgaClinAbrvs))
# {
#   fname <- paste("nationwidechildrens.org_clinical_patient_", allTcgaClinAbrvs[i], ".txt", sep="")
#   theUrl <- paste("https://tcga-data.nci.nih.gov/tcgafiles/ftp_auth/distro_ftpusers/anonymous/tumor/", allTcgaClinAbrvs[i] ,"/bcr/biotab/clin/nationwidechildrens.org_clinical_patient_", allTcgaClinAbrvs[i], ".txt", sep="")
#   download.file(theUrl, paste(clinicalFilesDir, fname, sep=""))
# }



test = read.csv("C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DatadataIn\\clinical\\nationwidechildrens.org_clinical_patient_brca.txt",sep="\t")

setwd("C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DatadataIn\\clinical")


allTcgaClinAbrvs <- c("acc", "blca", "brca", "cesc", "chol", "cntl", "coad", "dlbc", "esca", "fppp", "gbm", "hnsc", "kich", "kirc", "kirp", "laml", "lcml", "lgg", "lihc", "lnnh", "luad", "lusc", "meso", "misc", "ov", "paad", "pcpg", "prad", "read", "sarc", "skcm", "stad", "tgct", "thca", "thym", "ucec", "ucs", "uvm")
fname <- paste("nationwidechildrens.org_clinical_patient_",allTcgaClinAbrvs,".txt" , sep='')


###
test = read.csv("file:///C:/Users/raoki/Documents/GitHub/project_spring2019/DatadataIn/rnaSeq/gdac.broadinstitute.org_BRCA.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2015082100.0.0/BRCARN~1.txt", sep = "\t")





setwd("C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DatadataIn\\clinical")



allTcgaClinAbrvs <- c("acc", "blca", "brca", "cesc", "chol", "coad", "dlbc", "esca",  
                      "gbm", "hnsc", "kich", "kirc", "kirp", "laml", "lgg", "lihc", "luad", "lusc", 
                      "meso", "ov", "paad", "pcpg", "prad", "read", "sarc", "skcm", "stad", "tgct", 
                      "thca", "thym", "ucec", "ucs", "uvm")

allTcgaClinAbrvs_M <- c("acc", "blca", "brca", "cesc", "chol", "coad", "dlbc", "esca",  
                      "gbm", "hnsc", "kich", "kirc", "kirp", "laml", "lgg", "lihc", "luad", "lusc", 
                      "meso", "ov", "paad", "pcpg", "prad", "read", "sarc", "skcm", "stad", "tgct", 
                      "thca", "thym", "ucec", "ucs", "uvm")

fname <- paste("nationwidechildrens.org_clinical_patient_",allTcgaClinAbrvs,".txt" , sep='')
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
           "days_to_death" ) 

print(paste("columns:",length(cnames)))
for(i in 1:length(fname)){
  bd = read.csv(fname[i], sep = "\t") 
  print(paste(i, '-', length(intersect(names(bd),cnames))))
  #print(setdiff(cnames,names(bd)))
}

i = i + 1
bd = read.csv(fname[i], sep = "\t") 
names(bd) 


#### DOWNLOAD SOMATIC MUTATION 


diseaseAbbrvsForMuts <- c("ACC", "BLCA", "BRCA", "CHOL", "ESCA", "HNSC", "LGG", "LIHC", "LUSC", "MESO", "PAAD", "PRAD", "SARC", "SKCM", "TGCT", "UCS")
mutFilesDir <- paste(theRootDir, "\\mutation_data", sep="")
dir.create(mutFilesDir, showWarnings = FALSE) # make this directory if it doesn't exist.
for(i in 1:length(diseaseAbbrvsForMuts))
{
  mutationDataUrl <- paste("http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/", diseaseAbbrvsForMuts[i], "/20160128/gdac.broadinstitute.org_", diseaseAbbrvsForMuts[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz", sep="")
  fname <- paste("gdac.broadinstitute.org_", diseaseAbbrvsForMuts[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz", sep="")
  download.file(mutationDataUrl, paste(mutFilesDir, fname, sep=""))
}
thegzFiles <-  paste(mutFilesDir, dir(mutFilesDir), sep="")
sapply(thegzFiles, untar, exdir=mutFilesDir)
##      /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_ACC.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_BLCA.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_BRCA.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_CESC.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_CHOL.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_COAD.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
## /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_COADREAD.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_DLBC.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_ESCA.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##   /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_GBMLGG.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##      /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_GBM.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_HNSC.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_KICH.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##    /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_KIPAN.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_KIRC.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_KIRP.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_LAML.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##      /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_LGG.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_LIHC.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_LUAD.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_LUSC.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##       /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_OV.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_PAAD.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_PCPG.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_PRAD.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_READ.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_SARC.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_SKCM.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_STAD.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_STES.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_TGCT.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_THCA.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_THYM.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##     /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_UCEC.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##      /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_UCS.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0 
##      /mnt/data_scratch/finalData/dataIn/mutation_data/gdac.broadinstitute.org_UVM.Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz 
##                                                                                                                                       0
