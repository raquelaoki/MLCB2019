rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#AUTHOR: Raquel Aoki
#DATE: 2019/02/08
#Explorative script, getting to know the datasets first
#PROBLEM:
#There is two possible packages to use, Firebrowse or Cbioportal. 
#Check which one is better or more updated. 
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#Workdiretory 
setwd("~/GitHub/project_spring2019")

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#OPTION 1 - FIREBROWSE
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

#References: https://github.com/mariodeng/FirebrowseR/blob/master/vignettes/FirebrowseR.Rmd
#Install and load packages 
if (!require("FirebrowseR")) devtools::install_github("mariodeng/FirebrowseR")
require(FirebrowseR)

##Reading Data
cohorts = Metadata.Cohorts(format = "csv") # Download all available cohorts
cancer.Type = cohorts[grep("breast", cohorts$description, ignore.case = T), 1]

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
bd1 = do.call(rbind, brca.Pats)
dim(bd1)


#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#OPTION 2 - CBIOPORTAL
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

#Reference: http://www.cbioportal.org/rmatlab
#LInk above does not work, but help('cgdsr') works
if (!require("cgdsr")) install.packages("cgdsr")
require(cgdsr)

# Create CGDS object
mycgds = CGDS("http://www.cbioportal.org/")

# Test the CGDS endpoint URL using a few simple API tests
test(mycgds) 

# Get list of cancer studies at server
getCancerStudies(mycgds)

# Get available case lists (collection of samples) for a given cancer study  
mycancerstudy = getCancerStudies(mycgds)[33,1]
mycaselist = getCaseLists(mycgds,mycancerstudy)[1,1]

# Get available genetic profiles
mygeneticprofile = getGeneticProfiles(mycgds,mycancerstudy)[4,1]

# Get data slices for a specified list of genes, genetic profile and case list
getProfileData(mycgds,c('BRCA1','BRCA2'),mygeneticprofile,mycaselist)

# Get clinical data for the case list
bd2 = getClinicalData(mycgds,mycaselist)

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#COMPARING DATESETS
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

#checking the barcodes 
id1 = as.character(bd1[,1])
id2 = as.character(row.names(bd2))

id2 = gsub(".01", "", id2)
id2 = gsub(".", "-", id2, fixed = T)

id1 = id1[order(id1)]
id2 = id2[order(id2)]

head(id1); head(id2)
id1_2 = setdiff(id1, id2)

bd1$id = 0
for(i in 1:length(id1_2)){
  bd1$id[bd1$tcga_participant_barcode==id1_2[i]]=1
}
bd1_2 = subset(bd1, id==1)
dim(bd1_2); length(id1_2)
bd2$FORM_COMPLETION_DATE
summary(bd1$year_of_dcc_upload)


#CONCLUSION: Firebrowse seems to be more update and has more data than cbioportal. 
#most of the patients are presente in both studies. 