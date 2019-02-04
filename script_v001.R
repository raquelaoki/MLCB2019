rm(list=ls())
#Author: raoki
#References: https://github.com/mariodeng/FirebrowseR/blob/master/vignettes/FirebrowseR.Rmd
#Install and load packages 
if (!require("FirebrowseR")) devtools::install_github("mariodeng/FirebrowseR")
#require(XML)
require(FirebrowseR)

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

#brca.Pats = brca.Pats[ which(brca.Pats$vital_status == "dead"), ]

# Gene expression
#Number of participants is too much
diff.Exp.Genes = c("ESR1", "GATA3", "XBP1", "FOXA1", "ERBB2", "GRB7", "EGFR","FOXC1", "MYC")
all.Found = F
page.Counter = 1
mRNA.Exp = list()
page.Size = 2000 # using a bigger page size is faster
while(all.Found == F){
  mRNA.Exp[[page.Counter]] = Samples.mRNASeq(format = "csv",
                                             gene = diff.Exp.Genes,
                                             cohort = "BRCA",
                                             tcga_participant_barcode =
                                               brca.Pats$tcga_participant_barcode[1:200],
                                             page_size = page.Size,
                                             page = page.Counter)
  if(nrow(mRNA.Exp[[page.Counter]]) < page.Size)
    all.Found = T
  else
    page.Counter = page.Counter + 1
}
mRNA.Exp = do.call(rbind, mRNA.Exp)
dim(mRNA.Exp)


library(ggplot2)
p = ggplot(mRNA.Exp, aes(factor(gene), z.score))
p +
  geom_boxplot(aes(fill = factor(sample_type))) +
  # we drop some outlier, so plot looks nicer, this also causes the warning
  scale_y_continuous(limits = c(-1, 5)) +
  scale_fill_discrete(name = "Tissue")
