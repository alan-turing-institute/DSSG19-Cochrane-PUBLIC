#Purpose: R script for cleaning and loading papers data to SQL

#required libraries
library(RPostgreSQL)

#collect connection data from .pgpass file in home directory
pass_data = readLines(".pgpass")

#parse pass data
pass_data = unlist(strsplit(pass_data[grepl("cochrane", pass_data)], ":"))

#connect to SQL server
db_con = dbConnect(PostgreSQL(), 
                   user = pass_data[4],
                   password = pass_data[5],
                   dbname = pass_data[3],
                   host = "localhost")

#check to see if table exists and remove if so
if(dbExistsTable(db_con, "papers")){dbRemoveTable(db_con, "papers")}

#connect to data file
file_con = file("/data/raw/papers/RecordNotNullPMID.txt", open ="r")

#read in the first line and initiatlize resources
line = readLines(file_con, n=1)
list = list()
i = j = k = 0
start_time = Sys.time()

#read every line in file iteratively
while(length(line)>0){
  
  #create new table if header
  if(startsWith(line, "RecordID")){
    headers = unlist(strsplit(line, "\t"))
    df = data.frame(matrix(ncol=length(headers), nrow=0))
    names(df) = headers
    
    vars = rep("varchar",length(headers))
    names(vars) = headers
    
    dbCreateTable(db_con, "papers", vars)

  }else{
    #identify if the starting characters a record id
    start = substr(as.character(line), 1, 6)
    
    #if record id and no data in list, create new list with first instance
    if(!is.na(as.numeric(start)) & length(list) == 0){
      list[1] = line
    }else{
      
      #if not add to list to continue appending to record
      if(is.na(as.numeric(start)) & length(list) > 0){
        list = append(list, line)
      }else{
        
        #if new number, clean and organize observation
        if(!is.na(as.numeric(start)) & length(list) > 0){
          out = paste(unlist(list), collapse = " ")
          out = as.character(out)
          out = gsub("[\r\n]"," ", out)
          out[is.na(out)] = "NULL"
          out = paste(unlist(strsplit(out,"\t")), collapse="\t")
          valid = (length(unlist(strsplit(out,"\t"))) == 33)
          if(valid){
            out = as.data.frame(t(unlist(strsplit(out, "\t"))))
            names(out) = headers
            dbWriteTable(db_con, name = "papers", value = out, append=T, row.names = F)
            list=list()
            list[1]=line
            j = j + 1
          }else{
            write.table(x=out, file="/data/raw/papers/tmp_err.tsv",
                        append=T, row.names=F, col.names=F,
                        sep="\t", quote=F)
            list=list()
            list[1]=line
            k = k + 1
          }
        }
      }
    }
  }
line = readLines(file_con,n =1)
i = i + 1
if(i%%100000==0){
  print(paste("Line",i,"completed,",j,"rows added with",k,"errors.",round(Sys.time() - start_time, 2), "mins have elapsed."))
}
}

#close connections and unload resources 
dbDisconnect(db_con)
close(file_con)
gc()
