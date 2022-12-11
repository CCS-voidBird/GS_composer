#####GCTA vs RF####
# GCTA results reader

library(dplyr)
library(tidyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(forcats)
library(RColorBrewer)
options(digits=4)
save.image(file="E:/learning resource/OneDrive - The University of Queensland/PhD/HPC_Results/RF_blue/GctaVsRF.RData")
#load("E:/learning resource/OneDrive - The University of Queensland/PhD/HPC_Results/RF_blue/GctaVsRF.RData")

rf_SNP_importance <- read.csv("E:/learning resource/OneDrive - The University of Queensland/PhD/HPC_Results/RF_blue/2016to2017/models/RF_SNP_importances.csv",sep="\t")
rf_SNP_importance= rf_SNP_importance[rf_SNP_importance$Region == "all",]


#Plot SNP variance##
phenos = read.table("E:/learning resource/OneDrive - The University of Queensland/Sugarcane/Blue_phenotypes.txt",sep="\t",h=T)
a_genos <- read.table("E:/learning resource/OneDrive - The University of Queensland/PhD/HPC_Results/BLUPs/GCTA+MTG2/sugarcane_blue_all_A.ped",sep=" ")
d_genos <- read.table("E:/learning resource/OneDrive - The University of Queensland/PhD/HPC_Results/BLUPs/GCTA+MTG2/sugarcane_blue_all_D.ped",sep="\t")
ns = dim(phenos)[1]
ng = dim(a_genos)[2]
aMaf =apply(a_genos[,7:ng],2,sum,na.rm=T)/(2*ng)
dMaf = apply(d_genos[,7:ng],2,sum,na.rm=T)/(ng)
aVar <- apply(a_genos[,7:ng],2,var,na.rm=T)
dVar <- apply(d_genos[,7:ng],2,var,na.rm=T)
aavar <- aVar*aVar ##


vars = c(aVar,dVar,aavar)

#calculate additive PVE
get_pve <- function(maf,eff,seb2){
  upp <- 2*eff^2 * maf * (1-maf)
  bot <- upp + seb2 * (2*ns) * maf * (1-maf)
  res = upp / bot
  return(res)
}
#calculate additive and dominance PVE
get_adpve <- function(maf,eff,bot){
  upp <- 2*eff^2 * maf * (1-maf)
  bot <- upp + bot
  res = upp / bot
  return(res)
}


plot_relationship_gcta <- function(trait,grm,label,position) {
  grms=c("a","d","aa")
  print(trait)
  rf_importance = rf_SNP_importance
  rfsnps = rf_importance[(rf_importance$Trait == paste0(trait,"Blup") & rf_importance$Region == "all"),]
  rfsnps =array(rfsnps)[3:26088]
  y_index = which(c("tch","ccs","fibre") == tolower(trait))
  var.y = var(phenos[,1+y_index])
  
  rr_file_name = paste0("E:/learning resource/OneDrive - The University of Queensland/PhD/HPC_Results/BLUPs/gcta/1316to17/Blues_strict/",paste("16",tolower(trait),paste(grm,"snp.blp",sep="."),sep = "_"))
  snpdata = read.table(rr_file_name)
  rrsnp = snpdata[,3:2+which(grms == grm)] 
  a.se2 <- var.y - unlist(rrsnp[,1]^2) * unlist(aVar) 
  d.se2 <- var.y - unlist(rrsnp[,2]^2) * unlist(dVar)
  a.seb2 <- a.se2 / ((2*ng) * aMaf * (1-aMaf))
  d.seb2 <- d.se2 / ((2*ng) * dMaf * (1-dMaf))
  bot = 2*rrsnp[,1]^2 * aMaf * (1-aMaf) + 2*rrsnp[,2]^2 * dMaf * (1-dMaf) + a.seb2 * (2*ns) * aMaf * (1-aMaf) + d.seb2 * (2*ns) * dMaf * (1-dMaf)
  
  a.pve <- get_adpve(aMaf,rrsnp[,1],bot)
  d.pve <- get_adpve(dMaf,rrsnp[,2],bot)
  
  print("A PVE:")
  print(sum(a.pve,na.rm = T))
  
  print(head(rrsnp))
  print(dim(rfsnps))
  df = data.frame(log10(unlist(rfsnps)),log10(unlist(abs(a.pve))),log10(unlist(abs(d.pve))))
  colnames(df) <- c("RF","Additive","Dominance")
  title = trait
  
  print("linear fitting")
  shape_eff = lm(RF~Additive,data=df[df$RF != -Inf,])
  df_inf = df[df$RF != -Inf & !is.na(df$Dominance),]
  
  d = data.frame(unlist(rfsnps),unlist(abs(a.pve)),unlist(abs(d.pve)))
  head(d)
  colnames(d) <- c("RF","Additive","Dominance")
  d$sn = 1:dim(d[1])[1]
  print(cor(d$Additive,d$RF))
  print(cor(d$Dominance,d$RF))
  dm <- gather(df,key="Element",value="Log10_Importance",Additive,Dominance,RF)
  ddp <- ggplot(dm,aes(x=Log10_Importance,fill=Element))+geom_density(alpha=0.5)+
    theme(title=element_text(size = 25),
          axis.text = element_text(size=25),
          axis.title.x = element_blank(),
          legend.text = label,
          strip.text.x = element_text(size = 25),
          #axis.ticks.x = element_blank(),
          #axis.text.x = element_blank(),
          legend.position="none",legend.box = "horizontal")+guides(colour = guide_legend(nrow = 1))
  pa <- ggplot(df,aes(x=RF,y=Additive,colour=bias))+geom_point()+
    geom_smooth(method=lm,se=FALSE)+
    theme(title=element_text(size = 25),
          axis.text = element_text(size=25),
          legend.text = label,
          strip.text.x = element_text(size = 25),
          #axis.text.x = element_blank(),
          #axis.text.y = element_blank(),
          legend.position="none",legend.box = "horizontal")+
    annotate("text",y=max(df$Additive),x=median(df$RF),label=paste0("SnpEff Cor: ",sprintf("%.2f",cor(df_inf$RF,df_inf$Additive))),size=10)+
    labs(x="",y="")+ggtitle(title)+scale_color_gradientn(colours = c("red","grey","cyan")) #scale_color_gradient2(low = "cyan",mid = 'green',midpoint = -1,high = "red")
  pd <- ggplot(df,aes(x=RF,y=Dominance,colour=dbias))+geom_point()+
    geom_smooth(method=lm,se=FALSE)+
    theme(title=element_blank(),
          axis.text = element_text(size=25),
          legend.text = label,
          strip.text.x = element_text(size = 25),
          #axis.ticks.x = element_blank(),
          #axis.text.x = element_blank(),
          legend.position="none",legend.box = "horizontal")+
    annotate("text",y=max(df_inf$Dominance),x=median(df$RF),label=paste0("SnpEff Cor: ",sprintf("%.2f",cor(df_inf$RF,df_inf$Dominance))),size=10)+
    labs(x=NULL,y=NULL)+ggtitle(title)+scale_color_gradientn(colours = c("green","grey","cyan")) #scale_color_gradient2(low = "cyan",mid = 'green',midpoint = -1,high = "red")
  
  
  return(list(dm=dm,df=df,pa=pa,pd=pd,ddp=ddp))
}
ptch <- plot_relationship_gcta("TCH","d",element_text(size = 25),FALSE)

pccs <- plot_relationship_gcta("CCS","d",element_text(size = 25),FALSE)

pfibre <- plot_relationship_gcta("Fibre","d",element_text(size = 25),position="bottom")

library(cowplot)
lengend <- get_legend(
  ptch$ddp + 
    guides(color = guide_legend(nrow = 1)) +
    theme(legend.position = "bottom")
)

y.grob <- textGrob("Log10 GBLUP SNP_PVE", 
                   gp=gpar(fontface="bold", col="blue", fontsize=15), rot=90)

x.grob <- textGrob("Log10 RF SNP_Importance", 
                   gp=gpar(fontface="bold", col="blue", fontsize=15))


md <-  plot_grid(ptch$ddp, pccs$ddp,pfibre$ddp,nrow = 1, labels = LETTERS[7:9])


#md

x.d.grob <- textGrob("SNP_PVE or SNP_Importance", 
                     gp=gpar(fontface="bold", col="blue", fontsize=15))

y.d.grob <- textGrob("", 
                     gp=gpar(fontface="bold", col="blue", fontsize=15), rot=90)

mdg <- grid.arrange(arrangeGrob(md, left = y.d.grob, bottom = x.d.grob))
#md <- plot_grid(md,lengend,ncol = 1, rel_heights = c(1, .05))
final <- plot_grid(mpg,mdg,ncol = 1,rel_heights = c(2, 1))

final <- plot_grid(final,lengend,ncol = 1, rel_heights = c(1, .05))
final


d1 <- ggplot(ptch$dm,aes(x=Log10_Importance,fill=Element))+geom_density(alpha=0.5)+ggtitle("TCH")+
  theme(title=element_text(size = 25),
        axis.text = element_text(size=25),
        axis.title.x = element_blank(),
        legend.text = element_text(size = 25),
        strip.text.x = element_text(size = 25),
        axis.title.y = element_blank(),
        #axis.ticks.x = element_blank(),
        #axis.text.x = element_blank(),
        legend.position="none",legend.box = "horizontal")+guides(colour = guide_legend(nrow = 1))
d1

d2 <- ggplot(pccs$dm,aes(x=Log10_Importance,fill=Element))+geom_density(alpha=0.5)+ggtitle("   CCS")+
  theme(title=element_text(size = 25),
        axis.text = element_text(size=25),
        axis.title = element_blank(),
        legend.text = element_text(size = 25),
        strip.text.x = element_text(size = 25),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        legend.position="none",legend.box = "horizontal")+guides(colour = guide_legend(nrow = 1))
d2

d3 <- ggplot(pfibre$dm,aes(x=Log10_Importance,fill=Element))+geom_density(alpha=0.5)+ggtitle("   Fibre")+
  theme(title=element_text(size = 25),
        axis.text = element_text(size=25),
        axis.title = element_blank(),
        legend.text = element_text(size = 25),
        strip.text.x = element_text(size = 25),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        legend.position="none",legend.box = "horizontal")+guides(colour = guide_legend(nrow = 1))
d3

dall <- plot_grid(d1,d2,d3,nrow = 1)
dall
x.d.grob <- textGrob("Proportion of SNP PVE in GBLUP & SNP_Importance in RF", 
                     gp=gpar(fontface="bold", col="black", fontsize=15))
dall_b <- dall
df1 = plot_grid(dall_b,x.d.grob,ncol = 1, rel_heights = c(1, .1))
df1
df1 <- plot_grid(df1,nrow = 1)
lengend <- get_legend(
  ptch$ddp + 
    guides(color = guide_legend(nrow = 1)) +
    theme(legend.position = "bottom")
)



plot_relationship_variance <- function(trait,grm,label) {
  grms=c("a","d","aa")
  print(trait)
  rf_importance = rf_SNP_importance
  rfsnps = rf_importance[(rf_importance$Trait == paste0(trait,"Blup") & rf_importance$Region == "all"),]
  rfsnps =array(rfsnps)[3:26088]
  y_index = which(c("tch","ccs","fibre") == tolower(trait))
  var.y = var(phenos[,1+y_index])
  
  rr_file_name = paste0("E:/learning resource/OneDrive - The University of Queensland/PhD/HPC_Results/BLUPs/gcta/1316to17/Blues_strict/",paste("16",tolower(trait),paste(grm,"snp.blp",sep="."),sep = "_"))
  snpdata = read.table(rr_file_name)
  rrsnp = snpdata[,3:2+which(grms == grm)] 
  a.se2 <- var.y - unlist(rrsnp[,1]^2) * unlist(aVar) 
  d.se2 <- var.y - unlist(rrsnp[,2]^2) * unlist(dVar)
  a.seb2 <- a.se2 / ((2*ng) * aMaf * (1-aMaf))
  d.seb2 <- d.se2 / ((2*ng) * dMaf * (1-dMaf))
  bot = 2*rrsnp[,1]^2 * aMaf * (1-aMaf) + 2*rrsnp[,2]^2 * dMaf * (1-dMaf) + a.seb2 * (2*ns) * aMaf * (1-aMaf) + d.seb2 * (2*ns) * dMaf * (1-dMaf)
  
  a.pve <- get_adpve(aMaf,rrsnp[,1],bot)
  d.pve <- get_adpve(dMaf,rrsnp[,2],bot)
  
  print("A PVE:")
  print(sum(a.pve,na.rm = T))

  av = abs(rrsnp[,1]) * aVar
  dv = abs(rrsnp[,2]) * dVar
  
  av.w = av/sum(av)
  dv.w = dv/sum(dv)
  
  a.pve = av.w
  d.pve = dv.w
  
  print(head(rrsnp))
  print(dim(rfsnps))
  df = data.frame(log10(unlist(rfsnps)),log10(unlist(abs(a.pve))),log10(unlist(abs(d.pve))))
  #df = data.frame(unlist(rfsnps),unlist(abs(a.pve)),unlist(abs(d.pve)))
  colnames(df) <- c("RF","Additive","Dominance")
  #df$bias = (df$Additive / df$RF) #/ (mean(abs(df$rr)) / mean(df$rf))
  #df$dbias = (df$Dominance / df$RF)
  title = trait
  
  print("linear fitting")
  shape_eff = lm(RF~Additive,data=df[df$RF != -Inf,])
  #print(shape$coefficients)
  #print(coefficients(shape_var)[2])
  df_inf = df[df$RF != -Inf & !is.na(df$Dominance),]
  
  d = data.frame(unlist(rfsnps),unlist(abs(a.pve)),unlist(abs(d.pve)))
  head(d)
  colnames(d) <- c("RF","Additive","Dominance")
  d$sn = 1:dim(d[1])[1]
  print(cor(d$Additive,d$RF))
  print(cor(d$Dominance,d$RF))
  dm <- gather(df,key="Element",value="Log10_Importance",Additive,Dominance,RF)
  #dp <- ggplot(df[(df$RF != -Inf),],aes(x=RF))+geom_density()
  #dgp <- ggplot(df,aes(x=Additive))+geom_density()
  return(list(dm=dm,df=df))
}
ptch <- plot_relationship_variance("TCH","d",element_text(size = 25))
pccs <- plot_relationship_variance("CCS","d",element_text(size = 25))
pfibre <- plot_relationship_variance("Fibre","d",element_text(size = 25))

d1 <- ggplot(ptch$dm,aes(x=Log10_Importance,fill=Element))+geom_density(alpha=0.5)+ggtitle("TCH")+
  theme(
    title = element_blank(),
    axis.text = element_text(size=25),
    axis.title.x = element_blank(),
    legend.text = element_text(size = 25),
    strip.text.x = element_text(size = 25),
    axis.title.y = element_blank(),
    legend.position="none",legend.box = "horizontal")+guides(colour = guide_legend(nrow = 1))
d1

d2 <- ggplot(pccs$dm,aes(x=Log10_Importance,fill=Element))+geom_density(alpha=0.5)+ggtitle("   CCS")+
  theme(
    title = element_blank(),
    axis.text = element_text(size=25),
    axis.title = element_blank(),
    legend.text = element_text(size = 25),
    strip.text.x = element_text(size = 25),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    legend.position="none",legend.box = "horizontal")+guides(colour = guide_legend(nrow = 1))
d2

d3 <- ggplot(pfibre$dm,aes(x=Log10_Importance,fill=Element))+geom_density(alpha=0.5)+ggtitle("   Fibre")+
  theme(#title=element_text(size = 25),
    title = element_blank(),
    axis.text = element_text(size=25),
    axis.title = element_blank(),
    legend.text = element_text(size = 25),
    strip.text.x = element_text(size = 25),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    legend.position="none",legend.box = "horizontal")+guides(colour = guide_legend(nrow = 1))
d3

dall <- plot_grid(d1,d2,d3,nrow = 1)
dall <- plot_grid(dall,nrow = 1)
dall
x.d.grob <- textGrob("Proportion of SNP effect variance in GBLUP & SNP_Importance in RF", 
                     gp=gpar(fontface="bold", col="black", fontsize=15))
dall_b <- dall 
dfinal <- plot_grid(dall_b,x.d.grob,lengend,ncol = 1, rel_heights = c(1,.1, .1))
#dfinal <- grid.arrange(arrangeGrob(dfinal, bottom = x.d.grob))
dfinal

lengend <- get_legend(
  ptch$ddp + 
    guides(color = guide_legend(nrow = 1)) +
    theme(legend.position = "bottom")
)

dfinal <- plot_grid(df1,dall_b,x.d.grob,lengend,ncol = 1, rel_heights = c(1,0.8,.1, .1),label_size = 30)
dfinal




plot_relationship_models <- function(trait) {
  modelnames = c("A","AD","ADE")
  grms=c("a","d","aa")
  print(trait)
  rf_importance = rf_SNP_importance
  rfsnps = rf_importance[(rf_importance$Trait == paste0(trait,"Blup") & rf_importance$Region == "all"),]
  rfsnps =array(rfsnps)[3:26088]
  y_index = which(c("tch","ccs","fibre") == tolower(trait))
  var.y = var(phenos[,1+y_index])
  gvalue = data.frame(matrix(ncol = 3, nrow = 0))
  colnames(gvalue) <- c("trait","model","weight")
  for (i in 1:3){
    grm = grms[i]
    model = modelnames[i]
    rr_file_name = paste0("E:/learning resource/OneDrive - The University of Queensland/PhD/HPC_Results/BLUPs/gcta/1316to17/Blues_strict/",paste("16",tolower(trait),paste(grm,"snp.blp",sep="."),sep = "_"))
    snpdata = read.table(rr_file_name)
    lim = 2+i

    rrsnp = snpdata[,3:lim]
    gsnp = abs(rrsnp)
    if (i != 1)
    { for (v in 1:i){
      gsnp[,v] = gsnp[,v]*vars[v]
    }
      #sums = apply(gsnp,2,sum)
      for (j in 1:i){
        gsnp[,j] = gsnp[,j]/sum(gsnp[,j])
        weight = apply(gsnp,1,sum)
      }
    } else {sums = sum(gsnp)
    gsnp = gsnp * aVar
    weight = gsnp/sum(gsnp)}
    
    gsnp_ratio = data.frame(trait=trait,model=model,weight=weight)
    gvalue = rbind(gvalue,gsnp_ratio)
  }
  rfweight = data.frame(trait=trait,model="RF",weight=unlist(rfsnps))
  gvalue = rbind(gvalue,rfweight)
  gvalue$weight = log10(gvalue$weight)
  
  
  
  return(list(dm=gvalue))
}


ptch <- plot_relationship_models("TCH")

pccs <- plot_relationship_models("CCS")

pfibre <- plot_relationship_models("Fibre")

getPalette = colorRampPalette(brewer.pal(5, "Accent"))
d1 <- ggplot(ptch$dm,aes(x=weight,fill=model))+geom_density(alpha=0.5)+ggtitle("TCH")+labs(fill="Model")+
  theme(title=element_text(size = 25),
        axis.text = element_text(size=25),
        axis.title.x = element_blank(),
        legend.text = element_text(size = 25),
        strip.text.x = element_text(size = 25),
        axis.title.y = element_blank(),
        legend.position="none",legend.box = "horizontal")+guides(colour = guide_legend(nrow = 1))+
  scale_fill_manual(values = getPalette(4))+scale_color_manual(values = getPalette(4))
d1
lengend <- get_legend(
  d1 + 
    guides(fill = guide_legend(nrow = 1)) +
    theme(legend.position = "bottom")
)
d2 <- ggplot(pccs$dm,aes(x=weight,fill=model))+geom_density(alpha=0.5)+ggtitle("   CCS")+
  theme(title=element_text(size = 25),
        axis.text = element_text(size=25),
        axis.title = element_blank(),
        legend.text = element_text(size = 25),
        strip.text.x = element_text(size = 25),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        legend.position="none",legend.box = "horizontal")+guides(colour = guide_legend(nrow = 1))+
  scale_fill_manual(values = getPalette(4))+scale_color_manual(values = getPalette(4))
d2

d3 <- ggplot(pfibre$dm,aes(x=weight,fill=model))+geom_density(alpha=0.5)+ggtitle("   Fibre")+
  theme(title=element_text(size = 25),
        axis.text = element_text(size=25),
        axis.title = element_blank(),
        legend.text = element_text(size = 25),
        strip.text.x = element_text(size = 25),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        legend.position="none",legend.box = "horizontal")+guides(colour = guide_legend(nrow = 1))+
  scale_fill_manual(values = getPalette(4))+scale_color_manual(values = getPalette(4))
d3

dall <- plot_grid(d1,d2,d3,nrow = 1)
dall
x.d.grob <- textGrob("SNP weights in GBLUP & SNP_Importance in RF", 
                     gp=gpar(fontface="bold", col="black", fontsize=15))
dall_b <- dall #grid.arrange(arrangeGrob(dall, bottom = x.d.grob))
dfinal <- plot_grid(dall_b,x.d.grob,lengend,ncol = 1, rel_heights = c(1,.1, .1))
#dfinal <- grid.arrange(arrangeGrob(dfinal, bottom = x.d.grob))
dfinal

