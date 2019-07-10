rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#AUTHOR: Raquel Aoki
#DATE: 2019/02/26
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#Workdiretory 
setwd("C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019")

#Exploring the clinical data
#- Table S1. Detailed Information about TCGA-CDR and Analysis Results, 
#Related to Figures 1, 2, and S1-S5 and Tables 1-5 and S2-S4. TCGA-CDR 
#contains processed data for each individual cancer case including 
#essential clinicopathologic data and the four clinical outcome 
#endpoints of PFI, OS, DFI, and DSS. TCGA-CDR_Notes provides descriptions 
# of the data fields and the recommended use of the data fields. 
#- ExtraEndpoints contains additional clinical outcome endpoints. 
#- ExtraEndpoints_Notes provides descriptions of the data elements in Tab ExtraEndpoints. 
#- Table4_PHAssumptionTests shows the results of Cox PHs assumption tests 
#for models in Table 4, with notes. Table5_PHAssumptionTests shows
# the results of Cox PHs assumption tests for models in Table 5, with notes. 
#- TSS_Info shows information of TCGA tissue source sites. 
#Figure 2EFG_AdditionalInfo shows the results of additional analyses 
# of the example reported in Figures 2E-2G, including Cox PHs assumption 
#tests and competing risk assessment.


if (!require("openxlsx")) install.packages("openxlsx")

require("openxlsx")

#s2: read the original file on Excel. Clinical notes
#s3: extra endpoionts. I don't think I will use them. 
#s4: extra endpoionts notes. I don't think I will use them. 
#s6: PH Assumption tests: I don't think i will need it 
#s7: TSS info
#s8: additional info

tcga_clini <- read.xlsx("Data\\liu_s1_tcga_clinicalinformation.xlsx", sheet=1) 
tcga_tests <- read.xlsx("Data\\liu_s1_tcga_clinicalinformation.xlsx", sheet=5)[,-c(7,8,9)] 

#To identify metastase new_tumor_event_type
#There is several types, identify which ones I want to work with 
#Fixing labels - check with someone else if this is ok 

tcga_clini$new_tumor_event_type = as.character(tcga_clini$new_tumor_event_type)
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|[Not Available]']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='[Not Available]|[Not Available]']='[Not Available]'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|Distant Metastasis|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Locoregional Recurrence']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|Distant Metastasis|Regional lymph node']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|Locoregional Recurrence']='Distant Metastasis'

tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|New Primary Tumor']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Distant Metastasis|Regional lymph node']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Locoregional Recurrence|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Locoregional Recurrence|Locoregional Recurrence|Locoregional Recurrence|Distant Metastasis|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Regional lymph node|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Regional lymph node|Distant Metastasis|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Regional lymph node|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Locoregional Recurrence|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[tcga_clini$new_tumor_event_type=='Regional lymph node|Regional lymph node|Distant Metastasis|Regional lymph node|Distant Metastasis|Distant Metastasis']='Distant Metastasis'
tcga_clini$new_tumor_event_type[is.na(tcga_clini$new_tumor_event_type)] = 'Not obs new'

data.frame(table(tcga_clini$new_tumor_event_type))
s1 = subset(tcga_clini, type == "BRCA")

data.frame(table(s1$new_tumor_event_type))

#
#Firebroese seems not be good
#Comparing with FirebrowseR dataset 
if (!require("FirebrowseR")) devtools::install_github("mariodeng/FirebrowseR")
require(FirebrowseR)

 
# ##Reading Data
cohorts = Metadata.Cohorts(format = "csv") # Download all available cohorts
cancer.Type = cohorts[grep("breast", cohorts$description, ignore.case = T), 1]
# 
all.Received = F
page.Counter = 1
page.size = 150
brca.Pats = list()
while(all.Received == F){
  brca.Pats[[page.Counter]] = Samples.mRNASeq(format = 'csv', gene = 'BRCA',
  #brca.Pats[[page.Counter]] = Samples.Clinical(format = "csv",
                                               cohort = cancer.Type,
                                               page_size = page.size,
                                               page = page.Counter)
  if(page.Counter > 1)
    colnames(brca.Pats[[page.Counter]]) = colnames(brca.Pats[[page.Counter-1]])
  if(nrow(brca.Pats[[page.Counter]]) < page.size){
    all.Received = T
  } else{
    page.Counter = page.Counter + 1
  }
}
f1= do.call(rbind, brca.Pats)
head(f1)

data.frame(table(f1$distant_metastasis_present_ind2))



mRNA.Exp = Samples.mRNASeq(format = "csv",gene = c("PTEN", "RUNX1")) #,
                           tcga_participant_barcode = c("TCGA-GF-A4EO", "TCGA-AC-A2FG"))


mRNA.Exp = Samples.mRNASeq(format = "csv",tcga_participant_barcode = c("TCGA-GF-A4EO", "TCGA-AC-A2FG"))



#link with links
#http://gdac.broadinstitute.org/
#click cohort + data 
#http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/SKCM/20160128/
  
  
  
  
  
#Tests 
#bd1 = read.table('Data/SKCM.clin.merged.picked.txt', sep = '\t')  
#head(t(bd1))

bd2 = read.csv('Data/SKCM.merged_only_clinical_clin_format.txt', sep = '\t', header=T) #clinical info
bd4 = read.csv('Data/SKCM.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt',sep='\t') #normalized 

#removing some columns 
clinical = as.character(bd2$V1)













#------------------------- NEW CODE
#setwd("")

theRootDir <- "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data"
dir.create(paste(theRootDir, "dataIn/", sep=""), showWarnings = FALSE)
diseaseAbbrvs <- c("ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "COADREAD", "DLBC", "ESCA", "FPPP", "GBM", "GBMLGG", "HNSC", "KICH", "KIPAN", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD", "STES", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM")

missingAbrvsRnaSeq <- c(10, 31) # there is no RNA-seq data for "FPPP" or "STAD"
rnaSeqDiseaseAbbrvs <- diseaseAbbrvs[-missingAbrvsRnaSeq]
rnaSeqFilesDir <- paste(theRootDir, "dataIn/rnaSeq/", sep="")
dir.create(rnaSeqFilesDir, showWarnings = FALSE) # make this directory if it doesn't exist.
for(i in 1:length(rnaSeqDiseaseAbbrvs))
{
  fname <- paste("gdac.broadinstitute.org_", rnaSeqDiseaseAbbrvs[i], ".Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2015082100.0.0.tar.gz", sep="")
  download.file(paste("http://gdac.broadinstitute.org/runs/stddata__2015_08_21/data/", rnaSeqDiseaseAbbrvs[i], "/20150821/gdac.broadinstitute.org_", rnaSeqDiseaseAbbrvs[i], ".Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2015082100.0.0.tar.gz", sep=""), paste(rnaSeqFilesDir, fname, sep=""))
}

# Unzip the downloaded ".tar.gz" RNA-seq data! NB, this command has been tested in Linux. It may not work in Windows. If it does not work, please extract these files manually using software such as 7zip.
thegzFiles <-  paste(rnaSeqFilesDir, dir(rnaSeqFilesDir), sep="")
sapply(thegzFiles, untar, exdir=rnaSeqFilesDir)


clinicalFilesDir <- paste(theRootDir, "dataIn/clinical/", sep="")
dir.create(clinicalFilesDir, showWarnings = FALSE) # make this directory if it doesn't exist.

allTcgaClinAbrvs <- c("acc", "blca", "brca", "cesc", "chol", "cntl", "coad", "dlbc", "esca", "fppp", "gbm", "hnsc", "kich", "kirc", "kirp", "laml", "lcml", "lgg", "lihc", "lnnh", "luad", "lusc", "meso", "misc", "ov", "paad", "pcpg", "prad", "read", "sarc", "skcm", "stad", "tgct", "thca", "thym", "ucec", "ucs", "uvm")
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



