rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#AUTHOR: Raquel Aoki
#DATE: 2019/08
#check reference on email sent to Olga on March 2019 about download the data
#
#Notes: 
#1) The old scripts contain information about how to download/process RNA-seq, but we 
#decided to not use this datatype for now
#2) The merge between gene mutations and clinical information has a small intersection. 
#In some cancer types the intersection is 0 and for some others some patients with metastases are
#lost. The problem wasn't the date of the last data update, because I checked on the tcga official repo 
#and the data is also old there and the clinical information also incomplete. The main problem is that
#in the merge, many patients with mutation data have [incomplete information] on the variable 
#'new_tumor_event_dx_indicator', while the patients with NO or YES on this variable don't have the mutation data. 
#3) For some reason, the website https://portal.gdc.cancer.gov/repository don't open at SFU labs, but it works at my wifi. 
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

setwd("C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019")

#------------------------- CHANGE HERE TO DOWNLOAD DATA AGAIN
donwload_clinical = FALSE
donwload_mutation = FALSE
process_mutation = FALSE

theRootDir <- "C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\DataNew\\"
#Cancer types, MESO had to be removed for problems on the mutations part
diseaseAbbrvs <- c("ACC", "BLCA", "BRCA", "CHOL", "ESCA", "HNSC", "LGG", "LIHC", "LUSC", "PAAD", "PRAD", "SARC", "SKCM", "TGCT", "UCS")
diseaseAbbrvs_l <- c("acc", 'BRCA' ,"blca", "chol","esca", "hnsc", "lgg", "lihc", "lusc",  "paad", "prad", "sarc", "skcm",  "tgct", "ucs")


#------------------------ DOWNLOAD CLINICAL INFORMATION 
clinicalFilesDir <- paste(theRootDir, "clinical/", sep="")
dir.create(clinicalFilesDir, showWarnings = FALSE) # make this directory if it doesn't exist.

if(donwload_clinical){
  for(i in 1:length(diseaseAbbrvs)){
    fname <- paste("nationwidechildrens.org_clinical_patient_", allTcgaClinAbrvs[i], ".txt", sep="")
    theUrl <- paste("https://raw.github.com/paulgeeleher/tcgaData/master/nationwidechildrens.org_clinical_patient_", allTcgaClinAbrvs[i], ".txt", sep="")
    download.file(theUrl, paste(clinicalFilesDir, fname, sep=""))
  }
}


#------------------------ DOWNLOAD SOMATIC MUTATION 
mutFilesDir <- paste(theRootDir, "\\mutation_data", sep="")
dir.create(mutFilesDir, showWarnings = FALSE) # make this directory if it doesn't exist.
if(donwload_mutation){
  for(i in 1:length(diseaseAbbrvs)){
    mutationDataUrl <- paste("http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/", diseaseAbbrvsForMuts[i], "/20160128/gdac.broadinstitute.org_", diseaseAbbrvsForMuts[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz", sep="")
    fname <- paste("gdac.broadinstitute.org_", diseaseAbbrvsForMuts[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0.tar.gz", sep="")
    download.file(mutationDataUrl, paste(mutFilesDir, fname, sep=""))
  }
  thegzFiles <-  paste(mutFilesDir, dir(mutFilesDir), sep="\\")
  sapply(thegzFiles, untar, exdir=mutFilesDir)
}


#------------------------ CLINICAL INFORMATION DATA PROCESSING
#NOTE: columns have different names in each cancer type. These are more commom among them all
cnames = c("bcr_patient_barcode",'new_tumor_event_dx_indicator','abr') #"gender", "race" , "ethnicity", "tumor_status", "vital_status", #metastases is new_tumor_event


#Files names
fname1 <- paste(clinicalFilesDir,"nationwidechildrens.org_clinical_patient_",diseaseAbbrvs_l,".txt" , sep='')

#Rotine to read the files, select the important features, and bind in a unique dataset
i = 1
bd.aux = read.csv(fname1[i], sep = "\t") 
bd.aux$abr = diseaseAbbrvs[i]
bd.c = subset(bd.aux, select = cnames)

for(i in 2:length(fname1)){
  bd.aux = read.csv(fname1[i], sep = "\t", header = T) 
  bd.aux$abr = diseaseAbbrvs[i]
  bd.c = rbind(bd.c, subset(bd.aux, select = cnames))
}

bd.c = subset(bd.c, new_tumor_event_dx_indicator=="YES"|new_tumor_event_dx_indicator=="NO")
bd.c$new_tumor_event_dx_indicator  = as.character(bd.c$new_tumor_event_dx_indicator)

write.table(bd.c,paste(theRootDir,'tcga_cli.txt',sep=''), row.names = F, sep = ';')

#------------------------  MAF FILES / MUTATION DATA PROCESSING (time consuming)
#INSTALLING PACKAGES
if (!require("BiocManager"))
  install.packages("BiocManager")
if (!require("maftools"))
  BiocManager::install("maftools")
library(maftools)

exception = c("TCGA-P5-A5F6","TCGA-EJ-A7NG","TCGA-NA-A4QY") #codes with problems
#rotine to load data, manifest, for each patient will calculate the total number of mutations and merge with the other patients info
if(process_mutation){
  for(i in 1:length(diseaseAbbrvs)){
    mutationDataUrl <- paste(mutFilesDir,"\\gdac.broadinstitute.org_", diseaseAbbrvs[i],".Mutation_Packager_Calls.Level_3.2016012800.0.0", sep="")
    setwd(mutationDataUrl)
    manifest = read.table('MANIFEST.txt')
    for(j in 1:dim(manifest)[1]){
      barcode = substr(manifest$V2[j],0,12)
      cinfo = subset(bd.c, bcr_patient_barcode==barcode)
      if(sum(barcode==exception)==0){
        if(dim(cinfo)[1]==1){
          mutation = read.maf(maf = as.character(manifest$V2[j]), clinicalData = cinfo)
        }else{
          mutation = read.maf(maf = as.character(manifest$V2[j]))
        }
        if(j==1 & i==1){
          bd.m = getGeneSummary(mutation)
          bd.m = subset(bd.m, select = c('Hugo_Symbol','total'))
          names(bd.m)[2] = barcode
        }else{
          bd.m0 = getGeneSummary(mutation)
          bd.m0 = subset(bd.m0, select = c('Hugo_Symbol','total'))
          names(bd.m0)[2] = barcode
          bd.m = merge(bd.m,bd.m0,by = 'Hugo_Symbol',all=T)
        }
      }
    }
  }  
  #Remove NA for 0 
  bd.m[is.na(bd.m)] <- 0
  bd.m = bd.m[bd.m$Hugo_Symbol!='.']
  write.table(bd.m,paste(theRootDir,'tcga_mu.txt',sep=''), row.names = F, sep = ';')
  
}



#------------------------ MERGE CLINICAL INFORMATION AND MUTATION

bd.m = read.csv(paste(theRootDir, 'tcga_mu.txt',sep=''), header=T, sep=',')
bd.c = read.csv(paste(theRootDir, 'tcga_cli.txt',sep=''), header = T, sep=';')


#Transposing mutation dataset and fixing patient id (time consuming)
bd.m = t(bd.m)
bd.m = data.frame(rownames(bd.m),bd.m)
rownames(bd.m) = NULL
for( i in 1:dim(bd.m)[2]){
  names(bd.m)[i] = as.character(bd.m[1,i])
}
bd.m = bd.m[-1,]
names(bd.m)[1] = 'bcr_patient_barcode'
bd.m$bcr_patient_barcode = as.character(bd.m$bcr_patient_barcode)
bd.m$bcr_patient_barcode = gsub(pattern = '.', replacement = '-',bd.m$bcr_patient_barcode, fixed = T)
head(bd.m[,c(1:10)])


#Creating a variable indicator with the prediction value in the 0/1 format
bd.c$y = as.character(bd.c$new_tumor_event_dx_indicator)
bd.c$y[bd.c$y=="NO"] = 0 
bd.c$y[bd.c$y=="YES"] = 1 
bd.c = subset(bd.c, select = -c(new_tumor_event_dx_indicator))

#Merge: this part has problems, the intersection between the two datasets eliminate many good patients of our sampel
bd = merge(bd.c,bd.m, by = 'bcr_patient_barcode' , all=F)
bd$bcr_patient_barcode = as.character(bd$bcr_patient_barcode)
bd$abr = as.character(bd$abr)
head(bd[,c(1:10)])
table(bd$y,bd$abr)
prop.table(table(bd$y))

write.table(bd,paste(theRootDir,'tcga_train.txt',sep=''), row.names = F, sep = ';')

#------------------------ GENES SELECTION 
bd = read.table(paste(theRootDir,'tcga_train.txt',sep=''), header=T, sep = ';')
head(bd[,c(1:10)])
dim(bd)

#1) Eliminating genes mutated less than 15 times among all patients
el1 = colSums(bd[,-c(1,2,3)])
el1 = names(el1[el1<=15])
col1 = which(names(bd) %in% el1)
bd = bd[,-col1]
dim(bd)
write.table(bd,paste(theRootDir,'tcga_train_filted.txt',sep=''), row.names = F, sep = ';')


#2) Eliminating genes mutated less than 15 times among all patients
bd[,-c(1,2,3)][bd[,-c(1,2,3)]>=1]=1
el1 = colSums(bd[,-c(1,2,3)])
summary(el1)
el1 = names(el1[el1<=30])
col1 = which(names(bd) %in% el1)
bd = bd[,-col1]
dim(bd)
write.table(bd,paste(theRootDir,'tcga_train_binary.txt',sep=''), row.names = F, sep = ';')


#-------------------------- GENE EXPRESSION GENE SELECTION 
bd = read.table(paste(theRootDir,'tcga_rna_old.txt',sep=''), header=T, sep = ';')
bd = subset(bd, select = -c(patients2))
head(bd[,1:10])
dim(bd)

cl = read.table(paste(theRootDir,'tcga_cli_old.txt',sep=''), header=T, sep = ';')
cl = subset(cl, select = c(patients, new_tumor_event_dx_indicator))
names(cl)[2] = 'y'
cl$y = as.character(cl$y)
cl$y[cl$y=='NO'] = 0 
cl$y[cl$y=='YES'] = 1

bd1 = merge(cl,bd,by.x = 'patients',by.y = 'patients', all = F)
head(bd1[,1:10])

#check old code
#eliminate the ones with low variance 
require(resample)
var = colVars(bd1[,-c(1,2)])
var[is.na(var)]=0
#var = subset(var, !is.na(col))
datavar = data.frame(col = 1:dim(bd1)[2], var = c(100000,100000,var))
dim(datavar)
#datavar = subset(datavar, var>4.767836e+04 )
datavar = datavar[datavar$var>26604.77,]
dim(datavar)
bd1 = bd1[,c(datavar$col)]
head(bd1[,1:10])
dim(bd1)

order = c('patients','y',names(bd1))
order = unique(order)
bd1 = bd1[,order]

#eliminate the ones wich vales between 0 and 1 are not signnificantly different 
bdy0 = subset(bd1, y==0)
bdy1 = subset(bd1, y==1) 
pvalues = c()
for(i in 3:dim(bd1)[2]){
  pvalues[i-2] =  t.test(bdy0[,i],bdy1[,i])$p.value
  bd1[,i] = log(bd1[,i]+1)
}

#t.test:
#H0: y = x
#H1: y dif x 
#to reject the null H0 the pvalue must be <0.5
#i want to keep on my data the genes with y dif x, this 
#i want to filter small p values. 
ind = c(3:dim(bd1)[2])[pvalues<=0.00000001]
bd1 = bd1[,c(1,2,ind)]
head(bd1[,1:10])
dim(bd1)


write.table(bd1,paste(theRootDir,'tcga_train_gexpression.txt',sep=''), row.names = F, sep = ';')


