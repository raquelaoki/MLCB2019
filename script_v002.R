rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#AUTHOR: Raquel Aoki
#DATE: 2019/02/08
#Explorative cbioportal

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#Workdiretory 
setwd("~/GitHub/project_spring2019")


if (!require("FirebrowseR")) devtools::install_github("mariodeng/FirebrowseR")
if (!require("ggplot2")) install.packages("ggplot2")


#require(XML)
require(FirebrowseR)
require(ggplot2)

#Workdiretory 
setwd("~/GitHub/project_spring2019")

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
brca.Pats = do.call(rbind, brca.Pats)
dim(brca.Pats)
bd1 = brca.Pats

#brca.Pats = brca.Pats[ which(brca.Pats$vital_status == "dead"), ]

# Gene expression
#Number of participants is too much, selecting only these genes
# 
# diff.Exp.Genes = c("ESR1", "GATA3", "XBP1", "FOXA1", "ERBB2", "GRB7", "EGFR","FOXC1", "MYC")
# all.Found = F
# page.Counter = 1
# mRNA.Exp = list()
# page.Size = 2000 # using a bigger page size is faster
# while(all.Found == F){
#   mRNA.Exp[[page.Counter]] = Samples.mRNASeq(format = "csv",
#                                              gene = diff.Exp.Genes,
#                                              cohort = "BRCA",
#                                              tcga_participant_barcode =
#                                                brca.Pats$tcga_participant_barcode[1:200],
#                                              page_size = page.Size,
#                                              page = page.Counter)
#   if(nrow(mRNA.Exp[[page.Counter]]) < page.Size)
#     all.Found = T
#   else
#     page.Counter = page.Counter + 1
# }
# mRNA.Exp = do.call(rbind, mRNA.Exp)
# dim(mRNA.Exp)
# 
# 
# p = ggplot(mRNA.Exp, aes(factor(gene), z.score))
# p +
#   geom_boxplot(aes(fill = factor(sample_type))) +
#   # we drop some outlier, so plot looks nicer, this also causes the warning
#   scale_y_continuous(limits = c(-1, 5)) +
#   scale_fill_discrete(name = "Tissue")
# 


#Data from cbioportal
#http://www.cbioportal.org/rmatlab
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
myclinicaldata = getClinicalData(mycgds,mycaselist)
bd2 = myclinicaldata



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
