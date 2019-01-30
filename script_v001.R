#Author: raoki
#Install and load packages 
#if (!require("XML")) install.packages("XML")
#if (!require("methods")) install.packages("methods")
#require(XML)
#require(methods)

#Workdiretory 
setwd("~/GitHub/project_spring2019")

##Reading Data
#bd = xmlParse("Data/sample_submission_set.xml") 
  
# Exract the root node form the xml file.
#rootnode <- xmlRoot(bd)

# Find number of nodes in the root.
#rootsize <- xmlSize(rootnode)

# Print the result. adf
#print(rootsize)

# help file https://www.tutorialspoint.com/r/r_xml_files.htm 


###OMIM 
#bd1 = read.table('Data/OMIM_mim2gene.txt', header=T, sep="\t")
#bd2 = read.table('Data/OMIM_morbidmap.txt', header=T, sep="\t") #don't work
#bd3 = read.table('Data/OMIM_genemap2.txt', header=T, sep="\t") #don't read
#bd4 = read.table('Data/OMIM_mimTitles.txt', header=T, sep=";") #don't work


#UCI 
bd = read.table("Data/UCI_TCGA_data.csv", header=T,sep=',')
lb = read.table("Data/UCI_TCGA_labels.csv", header = T,sep=',')




