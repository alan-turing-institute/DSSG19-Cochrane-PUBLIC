#efficiency recall plots

#data
expand.grid("type"=c("Keep","Consider","Discard"),
            "model"=c("Baseline","Best","Citations"),
            "recall"=c(0.95,0.97,0.99)) %>%
  cbind("value"=c(0.009535, 0.070487, 0.919978,
                  0.009706, 0.062650, 0.927644,
                  0.010217, 0.060437, 0.929346,
                  0.009535, 0.107680, 0.882785,
                  0.009706, 0.096035, 0.894259,
                  0.010217, 0.093206, 0.896577,
                  0.01, 0.233, 0.758,
                  0.01, 0.218, 0.772,
                  0.01, 0.2202, 0.7695)) %>%
                  {merge(.,subset(.,model=="Baseline")%>%dplyr::select(recall, value, type), 
                         by=c("type","recall"), all.x=T)} -> df

#proportion to be manually reviewed


df%>%
  subset(type=="Consider") %>%
  ggplot(aes(x=factor(recall), y=value.x, group=model)) + 
  geom_line(aes(linetype=model, color=model), size=1.2) + 
  geom_point(aes(color=model), size=4) + 
  xlab("Recall") + 
  ylab("Proportion of papers to be manually reviewed") + 
  theme_bw()

#increase in efficiency 
df%>%
  subset(type=="Consider" & model!= "Baseline") %>%
  mutate(metric=(value.y-value.x)/value.y) %>%
  ggplot(aes(x=factor(recall), y=metric, group=model)) + 
  geom_line(aes(linetype=model, color=model), size=1.2) + 
  geom_point(aes(color=model), size=4) + 
  scale_y_continuous(labels = scales::percent)+
  labs(color="Model\ncompared to\nbaseline", linetype="Model\ncompared to\nbaseline")+
  xlab("Recall") + 
  ylab("Percent increase in efficiency compared to baseline") + 
  theme_bw()

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