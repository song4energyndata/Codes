rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
scriptpath <- getwd()
cat("\014")

library(plm)
library(lmtest)
library(sandwich)
library(multiwayvcov)

energyrnddat <- read.csv('energyrnd_rndpercapita.csv')
energyrnddat.former <- energyrnddat[energyrnddat$year<1998,] # 교토의정서 이전 기간
energyrnddat.later <- energyrnddat[energyrnddat$year>1997,] # 교토의정서 이후 기간


model.FE.former <- plm(log(ecperpop) ~ log(gdpperpop) + log(eprice) + log(hdd) + log(accrndperpop), data=energyrnddat.former, model="within")
summary(model.FE.former) # Fixed effect regression
coeftest(model.FE.former, vcov=vcovHC(model.FE.former, type="sss", cluster="group")) # 클러스터 표준오차
pwartest(model.FE.former)

model.FE.later <- plm(log(ecperpop) ~ log(gdpperpop) + log(eprice) + log(hdd) + log(accrndperpop), data=energyrnddat.later, model="within")
summary(model.FE.later)
coeftest(model.FE.later, vcov=vcovHC(model.FE.later, type="sss", cluster="group")) 
pwartest(model.FE.later)

model.FE.whole <- plm(log(ecperpop) ~ log(gdpperpop) + log(eprice) + log(hdd) + log(accrndperpop), data=energyrnddat, model="within")
summary(model.FE.whole)
coeftest(model.FE.whole, vcov=vcovHC(model.FE.whole, type="sss", cluster="group")) 
pwartest(model.FE.whole)


model.RE.former <- plm(log(ecperpop) ~ log(gdpperpop) + log(eprice) + log(hdd) + log(accrndperpop), data=energyrnddat.former, model="random")
phtest(model.FE.former, model.RE.former) # Hausman test

model.RE.later <- plm(log(ecperpop) ~ log(gdpperpop) + log(eprice) + log(hdd) + log(accrndperpop), data=energyrnddat.later, model="random")
phtest(model.FE.later, model.RE.later)


model.FD.former <- plm(log(ecperpop) ~ log(gdpperpop) + log(eprice) + log(hdd) + log(accrndperpop), data=energyrnddat.former, model="fd")
summary(model.FD.former) # First-order difference regression

model.FD.later <- plm(log(ecperpop) ~ log(gdpperpop) + log(eprice) + log(hdd) + log(accrndperpop), data=energyrnddat.later, model="fd")
summary(model.FD.later)

model.FD.whole <- plm(log(ecperpop) ~ log(gdpperpop) + log(eprice) + log(hdd) + log(accrndperpop), data=energyrnddat, model="fd")
summary(model.FD.whole)