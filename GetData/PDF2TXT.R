rm(list=ls())
setwd("~")

############################################
# @edufierro                               #
# Nov 2 - 2017                             #
# Project Scrapper - To Text               #
# Optimization Final Project               #                      
############################################

require(pdftools) # Require  "poppler", available from brew
require(stringr)

################
# Directiories #
################

# PDF's Location
dir1 <- "/Users/eduardofierro/Google Drive/TercerSemetre/Optimization/Project/Data/OriginalPDFs/"

# Output TXT location
dir2 <- "/Users/eduardofierro/Google Drive/TercerSemetre/Optimization/Project/Data/TXTsOriginal/"

# Location of Main.csv
dir3 <- "/Users/eduardofierro/Google Drive/TercerSemetre/Optimization/Project/Data/"

##############################
# Find files downloded twice #
##############################

# Some pdfs come in the form XXX (1).pdf and XXX.pdf. 

files <- list.files(dir1)
files <- files[grepl("(1)", files, fixed=T)] # There is none with format (2)
# Random and manually check some of those 

filesToCheck <- function(listFiles, n){	
	for (x in 1:n){
		print(files[round(runif(1, 1, length(files)))])
	}	
}

filesToCheck(files, 10)

# [1] "2599 (1).pdf"
# [1] "4916 (1).pdf"
# [1] "4918 (1).pdf"
# [1] "2599 (1).pdf"
# [1] "2896 (1).pdf"
# [1] "2893 (1).pdf"
# [1] "2887 (1).pdf"
# [1] "4922 (1).pdf"
# [1] "4919 (1).pdf"
# [1] "4914 (1).pdf"

# Yup, same --> IGNORE!!
rm(files, filesToCheck)

###################
# FROM PDF TO TXT #
###################

'%!in%' <- function(x,y)!('%in%'(x,y))

files <- list.files(dir1)
filesIgnore <- files[grepl("(1)", files, fixed=T)] 
files <- files[files %!in% filesIgnore]

pb <- txtProgressBar(min=1, max=length(files), style=3)
for (x in 1:length(files)){
	filename <- str_split(files[x], "[.]")[[1]][1]
	txt <- pdf_text(paste0(dir1, files[x]))
	writeLines(txt, paste0(dir2, filename, ".txt"))	    
	rm(txt, filename)
	setTxtProgressBar(pb, x)   
	
}
#rm(x)


filename <- str_split(files[x], "[.]")[[1]][1]
txt <- pdf_text(paste0(dir1, files[x]))
writeLines(txt, paste0(dir2, filename, ".txt"))

#################
# Missing files #
#################

# I got some errors, check which files are missing: 

pdfs_names <- list.files(dir1)
txt_names <- list.files(dir2)

main_data <- read.csv(paste0(dir3, "Main.csv"))

pdfs_names <- str_split(pdfs_names, "[.]")
pdfs_names <- do.call(rbind, pdfs_names)
pdfs_names <- as.numeric(pdfs_names[,1])

txt_names <- str_split(txt_names, "[.]")
txt_names <- do.call(rbind, txt_names)
txt_names <- as.numeric(txt_names[,1])

table(pdfs_names %in% txt_names) # Missing 52 TXT that should be in PDFs (This include the (1).pdf)

table(main_data$num %in% txt_names) # Missing 8 TXT that should be in main. WIll let it be. 