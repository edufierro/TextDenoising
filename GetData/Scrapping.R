rm(list=ls())
setwd("~")

############################################
# @edufierro                               #
# Nov 2 - 2017                             #
# Project Scrapper - SIL Mexico            #
# Optimization Final Project               #                      
############################################

require(XML)
require(plyr)
require(bitops)
require(RCurl)
require(stringr)
require(httr)
require(xlsx)
require(R2HTML)
require(readxl)

################
# Directiories #
################

# PDFS & DATA DIR ON LOCAL MACHINE
dir1 <- "/Users/eduardofierro/Google Drive/TercerSemetre/Optimization/Project/Data/OriginalPDFs/"

# MAIN DATA DIR
dir2 <- "/Users/eduardofierro/Google Drive/TercerSemetre/Optimization/Project/Data/"

########################
# First Table and URLs #
########################

# All initiatives of the LXIII Legislatura

main_table <- data.frame()

# 4361 files

pb <- txtProgressBar(min=1, max=291, style=3)

# CHECK THAT URL WORKS AND MAX PAGES! 
for (x in 1:400){
	url <- paste0("http://sil.gobernacion.gob.mx/Busquedas/Basica/ResultadosBusquedaBasica.php?SID=6db4616179044419288fb8247b854b2c&Origen=BB&Serial=6eef75d561bd4b550b9e3aa50c01d98d&Reg=5990&Paginas=15&pagina=", x)
	page.doc <- content(GET(url))
	page.doc <- htmlParse(page.doc)
	tables <- readHTMLTable(page.doc)
    tables <- as.data.frame(tables[[7]])
    tables[] <- lapply(tables, as.character)
    tables <- subset(tables, tables$No. != "")
    onclikcs <- xpathApply(page.doc, "//*/a", xmlGetAttr, "onclick")
    onclikcs <- do.call(rbind, onclikcs)
    onclikcs <- subset(onclikcs, str_detect(onclikcs[,1], "pp_ContenidoAsuntos"))
    onclikcs <- do.call(rbind, regmatches(onclikcs, gregexpr('"[^"]*"', onclikcs)))[,1]
    onclikcs <- gsub('\"', "" , onclikcs)
    tables <- cbind(tables, onclikcs)
    names(tables) <- c("num", "tipo", "denominacion", "clasificacion", "presentada", "fecha", "por", "partido_po", "legislatura", "turnado", "estatus", "tema", "onclicks")
    main_table <- rbind.fill(main_table, tables)	
    rm(tables, onclikcs, page.doc, url)
    setTxtProgressBar(pb, x)  
}
close(pb)
rm(pb, x)

main_table$onclicks <- as.character(main_table$onclicks)
write.csv(main_table, paste(dir2, "Main.csv", sep="/"), row.names=F)
write.xlsx(main_table, paste(dir2, "Main.xlsx", sep="/"), row.names=F)

########################
# By URL, download PDF #
########################

main_table <- read.csv(paste(dir2, "Main.csv", sep="/"))

### CODE PATCH IN CASE IT HAST TO BE RUN MULTIPLE TIMES: 
current_dir <- getwd()
setwd(dir1)
files <- list.files()
files <- gsub(".pdf", "", files)
setwd(current_dir)
contains <- as.character(main_table$num) %in% files

pb <- txtProgressBar(min=1, max=nrow(main_table), style=3)
for (x in 1:nrow(main_table)){
	if(contains[x]==FALSE){
		url <- paste0("http://sil.gobernacion.gob.mx", main_table$onclicks[x])
		page.doc <- content(GET(url))
		page.doc <- htmlParse(page.doc)
		pdf_url <- xpathSApply(page.doc, "//*/@href")[2]
		download.file(pdf_url, method="curl", destfile=paste0(dir1, main_table$num[x], ".pdf"),  mode="wb", quiet=TRUE)
		rm(page.docm,url)
	}
	setTxtProgressBar(pb, x)     
}
close(pb)
rm(x, pb)

