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




# 
# if (!require("FirebrowseR")) devtools::install_github("mariodeng/FirebrowseR")
# if (!require("ggplot2")) install.packages("ggplot2")
# 
# 
# #require(XML)
# require(FirebrowseR)
# require(ggplot2)
# 
# 
# ##Reading Data
# cohorts = Metadata.Cohorts(format = "csv") # Download all available cohorts
# cancer.Type = cohorts[grep("breast", cohorts$description, ignore.case = T), 1]
# 
# all.Received = F
# page.Counter = 1
# page.size = 150
# brca.Pats = list()
# while(all.Received == F){
#   brca.Pats[[page.Counter]] = Samples.Clinical(format = "csv",
#                                                cohort = cancer.Type,
#                                                page_size = page.size,
#                                                page = page.Counter)
#   if(page.Counter > 1)
#     colnames(brca.Pats[[page.Counter]]) = colnames(brca.Pats[[page.Counter-1]])
#   if(nrow(brca.Pats[[page.Counter]]) < page.size){
#     all.Received = T
#   } else{
#     page.Counter = page.Counter + 1
#   }
# }
# brca.Pats = do.call(rbind, brca.Pats)
# dim(brca.Pats)
# bd1 = brca.Pats
