rm(list=ls())
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

setwd("~\\Documents\\GitHub\\project")

donwload_clinical = FALSE
download_rna = FALSE
process_clinical = FALSE
process_rna = FALSE
genes_selection = FALSE

theRootDir <- "~\\Documents\\GitHub\\project\\data\\"
#Cancer types, MESO had to be removed for problems on the mutations part
diseaseAbbrvs <- c("ACC", "BLCA", "BRCA", "CHOL", "ESCA", "HNSC", "LGG", "LIHC", "LUSC", "PAAD", "PRAD", "SARC", "SKCM", "TGCT", "UCS")
diseaseAbbrvs_l <- c("acc", 'BRCA' ,"blca", "chol","esca", "hnsc", "lgg", "lihc", "lusc",  "paad", "prad", "sarc", "skcm",  "tgct", "ucs")


#------------------------ DOWNLOAD CLINICAL INFORMATION
#Downloading the clinical data using https://raw.github.com/paulgeeleher repo
if(donwload_clinical){
  clinicalFilesDir <- paste(theRootDir, "clinical/", sep="")
  dir.create(clinicalFilesDir, showWarnings = FALSE)

  for(i in 1:length(diseaseAbbrvs)){
    fname <- paste("nationwidechildrens.org_clinical_patient_", allTcgaClinAbrvs[i], ".txt", sep="")
    theUrl <- paste("https://raw.github.com/paulgeeleher/tcgaData/master/nationwidechildrens.org_clinical_patient_", allTcgaClinAbrvs[i], ".txt", sep="")
    download.file(theUrl, paste(clinicalFilesDir, fname, sep=""))
  }
}

#------------------------ DOWNLOAD RNA
#reference: http://gdac.broadinstitute.org/runs/stddata__2015_08_21/data/
if(download_rna){
  missingAbrvsRnaSeq <- c(10, 31) # there is no RNA-seq data for "FPPP" or "STAD"
  rnaSeqDiseaseAbbrvs <- diseaseAbbrvs[-missingAbrvsRnaSeq]
  rnaSeqFilesDir <- paste(theRootDir, "rnaSeq/", sep="")
  dir.create(rnaSeqFilesDir, showWarnings = FALSE) # make this directory if it doesn't exist.
  for(i in 1:length(rnaSeqDiseaseAbbrvs))
  {
    fname <- paste("gdac.broadinstitute.org_", rnaSeqDiseaseAbbrvs[i], ".Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2015082100.0.0.tar.gz", sep="")
    download.file(paste("http://gdac.broadinstitute.org/runs/stddata__2015_08_21/data/", rnaSeqDiseaseAbbrvs[i], "/20150821/gdac.broadinstitute.org_", rnaSeqDiseaseAbbrvs[i], ".Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.Level_3.2015082100.0.0.tar.gz", sep=""), paste(rnaSeqFilesDir, fname, sep=""))
  }

  # Unzip the downloaded ".tar.gz" RNA-seq data! NB, this command has been tested in Linux. It may not work in Windows. If it does not work, please extract these files manually using software such as 7zip.
  thegzFiles <-  paste(rnaSeqFilesDir, dir(rnaSeqFilesDir), sep="")
  sapply(thegzFiles, untar, exdir=rnaSeqFilesDir)
}

#------------------------ CLINICAL INFORMATION DATA PROCESSING
#NOTE: columns have different names in each cancer type. These are more commom among them all
if(process_clinical){
  cnames = c("bcr_patient_barcode",'new_tumor_event_dx_indicator','abr')
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

  write.table(bd.c,paste(theRootDir,'tcga_cli_old.txt',sep=''), row.names = F, sep = ';')
}

#------------------------ RNA DATA PROCESSING
if(process_rna){
  #creating function to extract the scaled_estimate per cancer type
  load_rna <- function(fname2){
    bd = read.csv(fname2, sep='\t',header = F)
    remove.c = c()
    patients = c()
    for(i in 2:dim(bd)[2]){
      if(as.character(bd[2,i])!='raw_count'){#scaled_estimate
        remove.c = c(remove.c,i)
      }else{
        patients = c(patients,as.character(bd[1,i]))
      }
    }

    #removing other columns
    bd.aux = data.frame(bd[,-remove.c])
    #removing first lines
    bd.aux = bd.aux[-c(1,2),]
    #organizing the columns names
    colname = strsplit(as.character(bd.aux$V1), split = "|", fixed = T)
    gene = c()
    gene_id = c()
    for(i in 1:length(colname)){
      gene[i] = colname[[i]][1]
      gene_id[i] = colname[[i]][2]
    }
    gene[gene=='?'] = paste('g',gene_id[gene=='?'],sep='')
    #transpose dataset
    bd.aux = data.frame(t(bd.aux[,-1]))
    names(bd.aux) = gene
    row.names(bd.aux)=NULL
    for(i in 1:dim(bd.aux)[2]){
      bd.aux[,i]= format(as.numeric(as.character(bd.aux[,i])), digits = 17)
    }
    bd.aux = data.frame(patients, bd.aux)
    return (bd.aux)
    }

  #Rotine to extract all RNA counts from the cancer types selected
  bd.e  = load_rna(fname2[1])
  print(paste('total ->', length(fname2)))
  tab[tab$Var1==diseaseAbbrvs[1],]
  for(i in 2:length(fname2)){
    print(paste(i,'-' ,diseaseAbbrvs[i]))
    bd.aux = load_rna(fname2[i])
    #checking if the columns names are the same and if they are in the same order
    print(paste('columns: ', dim(bd.e)[2], "---", sum(names(bd.aux)==names(bd.e))))
    if(sum(names(bd.aux)==names(bd.e))!=dim(bd.e)[2]){
      print('Error!')
    }
    bd.e = rbind(bd.e,bd.aux)
    write.csv(bd.e,paste(theRootDir,'tcga_rna_old.txt',sep=''), sep=';', row.names = F)
  }
}

#-------------------------- GENE EXPRESSION GENE SELECTION - keeping the driver genes
if(genes_selection){
  bd = read.table(paste(theRootDir,'tcga_rna_old.txt',sep=''), header=T, sep = ';')
  bd = subset(bd, select = -c(patients2))
  head(bd[,1:10])
  dim(bd)

  cl = read.table(paste(theRootDir,'tcga_cli_old.txt',sep=''), header=T, sep = ';')
  cl = subset(cl, select = c(patients, new_tumor_event_dx_indicator,abr))
  names(cl)[2] = 'y'
  cl$y = as.character(cl$y)
  cl$y[cl$y=='NO'] = 0
  cl$y[cl$y=='YES'] = 1

  bd1 = merge(cl,bd,by.x = 'patients',by.y = 'patients', all = F)
  head(bd1[,1:10])

  cgc = read.table(paste(theRootDir,'cancer_gene_census.csv',sep = ''),header=T, sep=',')[,c(1,5)]

  #eliminate the ones with low variance
  require(resample)
  exception = c(1,2,3)
  var = colVars(bd1[,-exception])
  var[is.na(var)]=0
  datavar = data.frame(col = 1:dim(bd1)[2], colname = names(bd1), var = c(rep(100000,length(exception)),var))

  #adding driver gene info
  datavar = merge(datavar, cgc, by.x='colname','Gene.Symbol',all.x=T)
  rows_eliminate = rownames(datavar)[datavar$var<500 & is.na(datavar$Tier)]#26604.77
  datavar = datavar[-as.numeric(as.character(rows_eliminate)),]

  bd1 = bd1[,c(datavar$col)]
  order = c('patients','y','abr',names(bd1))
  order = unique(order)
  bd1 = bd1[,order]
  head(bd1[,1:10])

  #eliminate the ones with values between 0 and 1 are not signnificantly different
  bdy0 = subset(bd1, y==0)
  bdy1 = subset(bd1, y==1)
  pvalues = rep(0,dim(bd1)[2])
  pvalues_ks = rep(0,dim(bd1)[2])
  for(i in (length(exception)+1):dim(bd1)[2]){
    bd1[,i] = log(bd1[,i]+1)
    pvalues[i] = wilcox.test(bdy0[,i],bdy1[,i])$p.value
  }

  #t.test:
  #H0: y = x
  #H1: y dif x
  #to reject the null H0 the pvalue must be <0.5
  #i want to keep on my data the genes with y dif x/small p values.
  datap = data.frame(col = 1:dim(bd1)[2], colname = names(bd1), pvalues = pvalues)
  datap = merge(datap, cgc, by.x='colname','Gene.Symbol',all.x=T)
  rows_eliminate =    rownames(datap)[datap$pvalues   >0.01 & is.na(datap$Tier)]
  datap = datap[-as.numeric(as.character(rows_eliminate)),]

  bd1 = bd1[,c(datap$col)]
  order = c('patients','y','abr',names(bd1))
  order = unique(order)
  bd1 = bd1[,order]

  #eliminate very correlated columns
  if(!file.exists(paste(theRootDir,'correlation_pairs.txt',sep=''))){
  i_ = c()
  j_ = c()
  i1 = length(exception)+1
  i2 = dim(bd1)[2]-1

  for(i in i1:i2){
    for(j in (i+1):(dim(bd1)[2])){
      if (abs(cor(bd1[,i],bd1[,j])) >0.70){
        i_ = c(i_,i)
        j_ = c(j_,j)
      }
    }
  }

  pairs = data.frame(i=i_,j=j_)
  #write.table(pairs,paste(theRootDir,'correlation_pairs.txt',sep=''), row.names = F, sep = ';')
  }else{
    pairs = read.table(paste(theRootDir,'correlation_pairs.txt',sep=''), header = T, sep = ';')
  }


  aux0 = pairs
  keep = c()
  remove = c()

  while(dim(aux0)[1]>0 ){
    aux00 = c(aux0$i,aux0$j)
    aux1 = data.frame(table(aux00))
    #subset(aux1, aux00 == 16245)
    aux1 = aux1[order(aux1$Freq,decreasing = TRUE),]

    keep = c(keep, as.numeric(as.character(aux1[1,1])))
    re0 = c(subset(aux0, i == as.character(aux1[1,1]))$j, subset(aux0, j == as.character(aux1[1,1]))$i)
    re0 = as.numeric(as.character(re0))
    remove = c(remove,re0)

    aux0 = subset(aux0, i!= as.character(aux1[1,1]))
    aux0 = subset(aux0, j!= as.character(aux1[1,1]))

    for(k in 1:length(re0)){
      aux0 = subset(aux0, i!=re0[k])
      aux0 = subset(aux0, j!=re0[k])
    }
  }


  datac = data.frame(col = 1:dim(bd1)[2], colname = names(bd1), rem = 0)
  datac = merge(datac, cgc, by.x='colname','Gene.Symbol',all.x=T)
  datac = datac[order(datac$col),]

  for(k in 1:length(remove)){
    if(is.na(datac[remove[k],]$Tier)){
      datac[remove[k],]$rem = 1
    }
    if(datac[remove[k],]$colname=='A1BG'){
      cat(k,remove[k])
    }
  }
  datac = subset(datac, rem==0)
  bd1 = bd1[,c(datac$col)]
  order = c('patients','y','abr',names(bd1))
  order = unique(order)
  bd1 = bd1[,order]
  head(bd1[,1:10])
  dim(bd1)


  #write.table(bd1,paste(theRootDir,'tcga_train_gexpression_cgc_7k.txt',sep=''), row.names = F, sep = ';')
}
