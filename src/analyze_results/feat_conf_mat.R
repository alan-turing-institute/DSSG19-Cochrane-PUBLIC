#scrip to create confusion matrix for features importance
library(ggplot2)
library(tidyverse)
library(ggrepel)

#create an object called "file" which contains the base data
top_x <- 10
discrete <- F


data <- read.csv(file)[-1]
data <- data[1:top_x,]



names <- names(data)

lapply(names, function(y)
  sapply(names, function(x) data[which(data[y]!=""),y] %in% data[which(data[x]!=""),x] %>% mean) %>% stack %>% mutate(ind2=y)) %>%
  {do.call("bind_rows", .)} %>%
  mutate(ind2 = factor(ind2, levels=rev(levels(ind)))) %>%
  mutate(values = values * 100) -> conf_mat_data

if(discrete){
  conf_mat_data %>%
    mutate(values=cut(values, c(0,0.25,0.5,0.75,1), include.lowest = T))
}
cutoff <- 15
ind_face <- ifelse(aggregate(.~ind, 
                 conf_mat_data[which(conf_mat_data$ind!=conf_mat_data$ind2),], 
                 FUN=max)$values >= cutoff, "bold","italic")

ind2_face <- ifelse(aggregate(.~ind2, 
                              conf_mat_data[which(conf_mat_data$ind!=conf_mat_data$ind2),], 
                              FUN=max)$values >= cutoff, "bold","italic")
ind_color <- ifelse(aggregate(.~ind, 
                             conf_mat_data[which(conf_mat_data$ind!=conf_mat_data$ind2),], 
                             FUN=max)$values >= cutoff, "black","gray")

ind2_color <- ifelse(aggregate(.~ind2, 
                              conf_mat_data[which(conf_mat_data$ind!=conf_mat_data$ind2),], 
                              FUN=max)$values >= cutoff, "black","gray")

ind_line_values <- levels(conf_mat_data$ind)[aggregate(.~ind, 
                         conf_mat_data[which(conf_mat_data$ind!=conf_mat_data$ind2),], 
                         FUN=max)$values >= cutoff]

ind2_line_values <- levels(conf_mat_data$ind2)[aggregate(.~ind2, 
                                                       conf_mat_data[which(conf_mat_data$ind!=conf_mat_data$ind2),], 
                                                       FUN=max)$values >= cutoff]



ggplot(conf_mat_data, aes(x=ind, y=ind2)) + 
  geom_tile(aes(fill=values)) + 
  scale_fill_distiller(palette = "BuPu", direction=1)+
  guides(fill=guide_colorbar("Percent\nFeature\nOverlap"))+
  # geom_vline(xintercept = which(rev(levels(conf_mat_data$ind2))%in%ind2_line_values),
  #            size=3, alpha = 0.3, color="gray")+
  # geom_hline(yintercept = which(rev(levels(conf_mat_data$ind))%in%ind_line_values),
  #            size=3, alpha = 0.3, color="gray")+
  geom_tile(data=conf_mat_data %>% subset(ind!=ind2 & values >=cutoff), 
            fill="yellow", color="black", size=1.05)+
  geom_label_repel(data=conf_mat_data %>% subset(ind!=ind2 & values >=cutoff),
             aes(label=paste0(values,"%"," overlap\n",ind,"-",ind2)),
             nudge_y=-5, nudge_x=3, force=2, fontface="bold")+
  theme(
    axis.text.x = element_text(angle=90, vjust=-0.000001,
                               face=ind_face, color=ind_color),
    axis.text.y = element_text(face=ind2_face, color=ind2_color),
    axis.title = element_blank()
  )
ggsave("conf.png", dpi=300, width=9, height=8, bg="transparent")
