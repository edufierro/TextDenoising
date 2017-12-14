# TextDenoising

Term Project: DS-GA 1013 Optimization-Based Data Analysis

Eduardo Fierro (eff254) & Raúl Delgado (rds491)

## Data scrapping

To download the data, using R (R version 3.3.3), you can run the code `GetData/Scrapping.R`. This code will output the pdf files found for all the bills, as well as an excel and csv file with the relevant information of each document. Each document will be name in a sequence from 1 to N, and uniquely identified with this table. To tranform the data to ".txt" files, run the code `GetData/PDF2TXT.R`. Both of this codes contain instructions to change local directiories at the top of each to point where the files should be saved in each case. Dependencies for these codes are:

```R
require(XML)
require(plyr)
require(bitops)
require(RCurl)
require(stringr)
require(httr)
require(xlsx)
require(R2HTML)
require(readxl)
require(pdftools)
```

Finally, the observations were randomly spit between train/validation/text (70%/15%/15%) using `GetData/DataSplitter.py`. Code runs in python3, with the requirements of numpy and os. 

## Glove Embeddings
