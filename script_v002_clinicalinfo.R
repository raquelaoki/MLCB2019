setwd("C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DatadataIn\\clinical")


allTcgaClinAbrvs <- c("acc", "blca", "brca", "cesc", "chol", "coad", "dlbc", "esca",  
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
		"metastatic_dx_confirmed_by" ,  
		"metastatic_dx_confirmed_by_other", 
		"metastatic_tumor_site" ,
   		"clinical_stage" ,  
		"days_to_birth" ,
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


bd = read.csv(fname[i], sep = "\t") 
