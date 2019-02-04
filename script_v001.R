#Author: raoki
#Install and load packages 
if (!require("FirebrowseR")) devtools::install_github("mariodeng/FirebrowseR")
#require(XML)
require(FirebrowseR)

#Workdiretory 
setwd("~/GitHub/project_spring2019")

##Reading Data
cohorts = Metadata.Cohorts(format = "csv") # Download all available cohorts




