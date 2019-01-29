#Author: raoki
#Install and load packages 
if (!require("XML")) install.packages("XML")
if (!require("methods")) install.packages("methods")
require(XML)
require(methods)

#Workdiretory 
setwd("~/GitHub/project_spring2019")

#Reading Data
bd = xmlParse("Data/sample_submission_set.xml") 
  
# Exract the root node form the xml file.
rootnode <- xmlRoot(bd)

# Find number of nodes in the root.
rootsize <- xmlSize(rootnode)

# Print the result. adf
print(rootsize)

# help file https://www.tutorialspoint.com/r/r_xml_files.htm 

