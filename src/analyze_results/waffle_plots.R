library(ggplot2)
library(tidyverse)
library(ggsci)
library(scales)

create_df <- function(named_list, facet){
  
  #initialize data frame
  df <- expand.grid(y = 1:10, x = 1:10)
  
  #update df 
  names <- names(named_list)
  var <- c(rep(names[1],named_list[1]),
           rep(names[2],named_list[2]),
           rep(names[3],named_list[3]))
  categ_table <- round(table(var) * ((10*10)/(length(var))))[names]
  df$category <- factor(rep(names(categ_table), categ_table),levels=names)
  df$facet <- facet
  
  return(df)
}

df <- rbind(
  create_df(list(" Keep "=2," Consider "=54," Discard "=44), "Baseline"),
  create_df(list(" Keep "=1," Consider "=22," Discard "=77), "Best")
  )

df <- create_df(list(" Keep "=1," Consider "=22," Discard "=77), "Best")

## Plot
ggplot(df, aes(x = x, y = y)) + 
  geom_tile(aes(fill=category),color = "white", size=1, alpha=0.8) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0), trans = 'reverse') +
  scale_fill_manual(values=c("#33a02c","#fdae61","#e41a1c")) +
  facet_grid(facet~.)+
  labs(title=paste0("Average workload for ",nrow(df)," papers"), 
       subtitle=paste0("At a recall of 99%, ",
                       nrow(subset(df, category==" Discard ")), 
                       " papers can be discarded on average")) + 
  theme(plot.title = element_text(size = 20, hjust=0.5, face="bold"),
        plot.subtitle = element_text(hjust=0.5, size=16),
        axis.text = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank(),
        legend.title = element_blank(),
        legend.position = "bottom",
        legend.text = element_text(size=16))



## great bar plots for keep consider discard 

expand.grid(
  "fill"=c("Keep","Consider","Discard","Unmeasured"),
  "type"=c("Baseline -\nHuman","Baseline -\nMachine","Best","Best +\nCitations")
  )%>%
  data.frame %>%
  cbind("value"=c(0,0,0,100,
                  1,23.3,75.8,0,
                  1,21.8,77.2,0,
                  1,22.03,76.9,0)) %>%
  cbind("pos"=c(NA,NA,NA,NA,
                25,10,60,NA,
                25,10,60,NA,
                25,10,60,NA)) %>%
  ggplot(aes(x=factor(type, levels=rev(c("Baseline -\nHuman","Baseline -\nMachine","Best","Best +\nCitations"))), y=value))+
  geom_histogram(aes(fill=factor(fill, levels=rev(c("Consider","Keep","Discard","Unmeasured")))), 
                 stat="identity", color="black",size=1.3) + 
  geom_label(aes(y=pos, label = paste0(value,"%")), size=7, fontface="bold")+
  scale_fill_manual(values=c("#6F99ADFF","#BC3C29FF","#E18727FF", "#0072B5FF"))+
  coord_flip() + 
  theme_classic()+
  guides(fill=guide_legend(reverse = T)) +
  theme(legend.position="bottom", legend.direction="horizontal",
        legend.title = element_blank(),
        axis.title = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size=18, face="bold", angle=90, vjust=-0.5, hjust=0.5),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.text = element_text(size=18))



ggsave("barplot.png", dpi=300, width=8, height=5, bg="transparent")



#95: 
#Baseline: 0.009535, 0.070487, 0.919978
#Best: 0.009706, 0.062650, 0.927644
#Citations: 0.010217, 0.060437, 0.929346

#97
#Baseline: 0.009535, 0.107680, 0.882785
#Best: 0.009706, 0.096035, 0.894259
#Citations: 0.010217, 0.093206, 0.896577

#99
#Baseline: 0.01, 0.233, 0.758
#Best: 0.01, 0.218, 0.772
#Citations: 0.01, 0.2202, 0.7695