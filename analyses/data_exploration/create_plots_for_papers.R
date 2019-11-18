#R script for basic EDA plots 

#dependencies 
library(tidyverse)
library(RPostgreSQL)
library(ggplot2)
library(scales)
library(treemap)

#paths
out_file = "/data/figs/"

#SQL credentials
pass_data = readLines(".pgpass")
pass_data = unlist(strsplit(pass_data[grepl("cochrane", pass_data)], ":"))

#connect to SQL server
db_con = dbConnect(PostgreSQL(), 
                   user = pass_data[4],
                   password = pass_data[5],
                   dbname = pass_data[3],
                   host = "localhost")


#How often do papers show up in our data
dbGetQuery(db_con, 
"select \"RecordID\", count(*) as n from raw.papers
group by \"RecordID\" order by n desc;") %>%
  ggplot(aes(x=log10(n))) + 
  geom_histogram(color="darkred", fill="red") + 
  scale_y_continuous("Count", labels=scales::comma) +
  theme_bw() + 
  ggtitle("Log_10 Number of times paper ID in DB")
ggsave("/data/figs/eda/num_papers_in_db.png")

#How often do papers show up in reviews 
dbGetQuery(db_con, 
           "select \"RecordID\", count(*) as n from raw.papers
where \"CN\"!=\'NULL\'
group by \"RecordID\" order by n desc;") %>%
  ggplot(aes(x=n)) + 
  geom_histogram(color="darkblue", fill="lightblue") + 
  scale_y_log10("Count", labels=scales::comma) +
  theme_bw() + 
  ggtitle("Number of times paper in review")
ggsave("/data/figs/eda/num_papers_in_review.png")

#How many records are included in reviews?
dbGetQuery(db_con, 
           "select \"CN\", count(*) as n
from raw.papers
where \"CN\"!=\'NULL\'
group by 1 order by 2 desc;") -> x
ggplot(data=x,aes(x=n)) + 
  geom_histogram(color="black", fill="coral1") + 
  scale_y_log10("Count", labels=scales::comma) +
  theme_bw() + 
  ggtitle("Distribution of papers in reviews")
ggsave("/data/figs/eda/papers_used_in_reviews.png")

#Distribution of paper languages
dbGetQuery(db_con,
           "select \"LA\", count(*) as n from raw.papers
group by 1 order by n desc;"
) -> x
x %>%
  mutate(lang=case_when(
    grepl("eng",tolower(as.character(LA))) ~ "English",
    grepl("chi",tolower(as.character(LA))) ~ "Chinese",
    grepl("ger",tolower(as.character(LA))) ~ "German",
    grepl("null",tolower(as.character(LA))) ~ "Unknown",
    TRUE ~ "Other"
  )) %>% 
  {aggregate(.$n, by=list(.$lang), FUN=sum)} %>%
  {pie(.$x,paste(.$Group.1, round((.$x/sum(.$x))*100,2),"%"), main="Distribution of languages by record")}
