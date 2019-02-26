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
  brca.Pats[[page.Counter]] = Samples.Clinical(format = "csv",
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
