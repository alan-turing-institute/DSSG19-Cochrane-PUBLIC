library(tidyverse)
library(ggplot2)

best <- read.csv('/path/to/results/results_overall.csv') %>%
  dplyr::select(label,recall,precision_at_recall) %>% 
  group_by(label, recall) %>% 
  filter(precision_at_recall == max(precision_at_recall)) %>% 
  arrange(label, recall, precision_at_recall) %>% unique
comp <- read.csv('/path/to/results/results_1.csv') %>%
  dplyr::select(label,recall,precision_at_recall)
ref <- read.csv('/path/to/results/ref.csv') %>%
  dplyr::select(label, n_revs, n_papers) %>%
  unique

heatmap_data <- merge(best,comp, by=c("label","recall")) %>%
  merge(ref, by="label") %>%
  mutate(metric=precision_at_recall.x-precision_at_recall.y) %>%
  mutate(disc_metric = cut(metric, seq(0,0.4,0.1)))

#heatmap continuous
ggplot(data=heatmap_data, aes(x=recall,
                              y=factor(label,levels=ref[order(ref$n_papers),"label"])))+
  geom_tile(aes(fill=metric),colour="black",size=0.2)+
  coord_cartesian(xlim = c(0.9, 0.99))+
  #scale_fill_viridis_c()+
  scale_fill_distiller(palette = "Spectral", direction=-1)+
  xlab("Recall") + ylab("Review Groups") +
  guides(fill=guide_colourbar(title="Increase \n in Precision\n compared to \n baseline"))+
  labs(title="Overall Performance - RGs sorted by # of Papers")+
  theme_grey(base_size=10)

#discrete metric
ggplot(data=heatmap_data, aes(x=recall,
                              y=factor(label,levels=ref[order(ref$n_papers),"label"])))+
  geom_tile(aes(fill=disc_metric),colour="black",size=0.2)+
  coord_cartesian(xlim = c(0.9, 0.99))+
  scale_fill_manual(values = c("#f0f9e8", "#bae4bc", "#7bccc4", "#2b8cbe")%>% rev, 
                    limits = rev(unique(heatmap_data$disc_metric)) %>% as.character,
                    na.translate=T,na.value="gray",
                    labels=c(as.character(rev(unique(heatmap_data$disc_metric))[1:4]),"No \ndifference")) +
  xlab("Recall") + ylab("Review Groups") +
  guides(fill=guide_legend(title="Increase\nin Precision\ncompared to\nbaseline"))+
  labs(title="Overall Performance - RGs sorted by # of Papers")+
  theme_grey(base_size=10)

#heatmap absolute
ggplot(data=best, aes(x=recall,
                      y=factor(label,levels=ref[order(ref$n_papers),"label"])))+
  geom_tile(aes(fill=precision_at_recall),colour="black",size=0.2)+
  coord_cartesian(xlim = c(0.9, 0.99))+
  scale_fill_viridis_c()+
  #scale_fill_distiller(palette = "Spectral", direction=-1)+
  xlab("Recall") + ylab("Review Groups") +
  guides(fill=guide_colourbar(title="Precision"))+
  labs(title="Precision - RGs sorted by # of Papers")+
  theme_grey(base_size=10)
