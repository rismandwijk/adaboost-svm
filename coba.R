message("Packages")
library(e1071)
library(caret)
library(tidyverse)
message("Data")
data_churn<-read.csv("https://raw.githubusercontent.com/rismandwij/Data/main/IT_customer_churn.csv")
data_churn <- data_churn %>% drop_na()
message("Preprocessing")
data_churn1<-data_churn
data_churn1$gender<-as.factor(ifelse(data_churn1$gender=="Male",1,0))
data_churn1$Partner<-as.factor(ifelse(data_churn1$Partner=="Yes",1,0))
data_churn1$Dependents<-as.factor(ifelse(data_churn1$Dependents=="Yes",1,0))
data_churn1$PhoneService<-as.factor(ifelse(data_churn1$PhoneService=="Yes",1,0))
data_churn1$MultipleLines<-as.factor(ifelse(data_churn1$MultipleLines=="Yes",2,ifelse(data_churn1$MultipleLines=="No",1,0)))
data_churn1$InternetService<-as.factor(ifelse(data_churn1$InternetService=="Fiber optic",2,ifelse(data_churn1$MultipleLines=="DSL",1,0)))
data_churn1$OnlineSecurity<-as.factor(ifelse(data_churn1$OnlineSecurity=="Yes",2,ifelse(data_churn1$OnlineSecurity=="No",1,0)))
data_churn1$OnlineBackup<-as.factor(ifelse(data_churn1$OnlineBackup=="Yes",2,ifelse(data_churn1$OnlineBackup=="No",1,0)))
data_churn1$DeviceProtection<-as.factor(ifelse(data_churn1$DeviceProtection=="Yes",2,ifelse(data_churn1$DeviceProtection=="No",1,0)))
data_churn1$TechSupport<-as.factor(ifelse(data_churn1$TechSupport=="Yes",2,ifelse(data_churn1$TechSupport=="No",1,0)))
data_churn1$StreamingTV<-as.factor(ifelse(data_churn1$StreamingTV=="Yes",2,ifelse(data_churn1$StreamingTV=="No",1,0)))
data_churn1$StreamingMovies<-as.factor(ifelse(data_churn1$StreamingMovies=="Yes",2,ifelse(data_churn1$StreamingMovies=="No",1,0)))
data_churn1$Contract<-ifelse(data_churn1$Contract=="Two year",2,ifelse(data_churn1$Contract=="One year",1,1/12))
data_churn1$PaperlessBilling<-as.factor(ifelse(data_churn1$PaperlessBilling=="Yes",1,0))
data_churn1$PaymentMethod<-as.factor(ifelse(data_churn1$PaymentMethod=="Bank transfer (automatic)"|data_churn1$PaymentMethod=="Credit card (automatic)",1,0))
data_churn1$Churn<-as.factor(ifelse(data_churn1$Churn=="Yes",1,0))
message("Partition")
set.seed(1037)
kelas0<-subset(data_churn1,data_churn1$Churn==0)
kelas1<-subset(data_churn1,data_churn1$Churn==1)
k0_p<-sample(1:nrow(kelas0), size = round(0.9*nrow(kelas0)), replace=FALSE)
k1_p<-sample(1:nrow(kelas1), size = round(0.9*nrow(kelas1)), replace=FALSE)
latih<-rbind(kelas0[k0_p,],kelas1[k1_p,])
uji<-rbind(kelas0[-k0_p,],kelas1[-k1_p,])
f_data_asli<-as.vector(table(data_churn1$Churn))
f_data_latih<-as.vector(table(latih$Churn))
f_data_uji<-as.vector(table(uji$Churn))
nama_data<-c("Data Asli","Data Latih","Data Uji")
partisi_data<-data.frame(nama_data,rbind(f_data_asli,f_data_latih,f_data_uji))
rownames(partisi_data)<-NULL
colnames(partisi_data)<-c("Data","Kelas_0","Kelas_1")

message("Adaboost SVM Linear")
print("Adaboost SVM Linear")
stratified.cv=function(data,nfolds){
  folds <- createFolds(factor(data[,ncol(data)]), k = nfolds, list = FALSE)
  data.train=vector("list",nfolds)
  data.test=vector("list",nfolds)
  for (i in 1:nfolds){
    fold<-which(folds==i,arr.ind = TRUE)
    data.train[[i]]<-data[-fold,]
    data.test[[i]]<-data[fold,]
  }
  data=list(data.train,data.test)
  #list [[1]][[1:5]] == data train dan list [[2]][[5:10]] == data.test
  return(data)
}
adaboostori=function(X,y,X.test,y.test,iterasi,kernel,gamma=NULL,cost){
  #kernel yang digunakan hanya rbf dan linear
  if(!is.matrix(X)) X=as.matrix(X)
  if(!is.matrix(X)) X.test=as.matrix(X.test)
  n=nrow(X)
  final.test <- rep(0, length(y.test))
  bobot=rep(1/n,n)
  final.pred=list()
  error=c()
  a=c()
  Sign=function(x,kelas.positif,kelas.negatif){
    tanda=ifelse(x>=0,kelas.positif,kelas.negatif)
    return(tanda)
  }
  for (i in 1:iterasi){
    if(i == 1) { samp=sample(nrow(X), nrow(X), replace = FALSE) }
    # sampling dengan pengembalian pada iterasi ke dua dan seterusnya
    else if(i != 1) { samp=sample(nrow(X), nrow(X), replace = TRUE, prob = bobot) }
    # membuat trainning set baru
    X.train=X[samp,]
    row.names(X.train)=NULL
    y.train=y[samp]
    if (length(y.train[y.train==-1])==0|length(y.train[y.train==1])==0) {
      cat("y train hanya berisi satu kelas","\n")
      a[i]=0
      final.pred[[i]]=matrix(0,ncol=1,nrow=length(y.test))
      break}
    row.names(y.train)=NULL
    #training svm
    if (kernel=="linear"){
      weight.svm=svm(x=X.train,y=y.train,scale=F,kernel=kernel,cost=cost)
    }
    else {
      weight.svm=svm(x=X.train,y=y.train,scale=F,kernel=kernel,gamma=gamma,cost=cost)
    }
    #prediksi seluruh data trainning kemudian menghitung error
    pred=predict(weight.svm,X)
    error[i]=sum(bobot*ifelse(pred!=y,1,0))/sum(bobot)
    if (error[i]<=0.000001) { print("iterasi berhenti")
      cat("iterasi=",i,"\n")
      cat("error=",error[i],"\n")
      a[i]=0
      final.pred[[i]]=matrix(0,ncol=1,nrow=length(y.test))
      break }
    else if (error[i]>=0.49999) {
      a[i]=0 ;bobot=rep(1/n,n)}
    else {
      #menghitung bobot classifier
      a[i]=(1/2)*log((1-error[i])/error[i])
      #mengupdate bobot pengamatan
      bobot=(bobot*exp(-a[i]*ifelse(pred!=y,-1,1)))/sum(bobot*exp(-a[i]*ifelse(pred!=y,-1,1)))}
    # mengklasifikasikan data testing
    final.pred[[i]]=attr(predict(weight.svm,X.test,decision.values=TRUE),"decision.values")
    if(colnames(final.pred[[i]])=="-1/1"){
      final.pred[[i]]=-final.pred[[i]]}
    else { final.pred[[i]]=final.pred[[i]]}
    a=a/sum(a)
    if(is.nan(a[1])) {a=rep(0,length(a))}
    a=as.list(a)
    # menggabungkan hasil klasifikasi (final classifier)
    dd=lapply(1:length(final.pred),function(i){final.pred[[i]]*a[[i]]})
    final.test=do.call("cbind",dd)
    final.test=rowSums(final.test)
    prediksi.kelas=Sign(final.test,1,-1)
    hasil=list(bobot.final=bobot,fit.y=final.test,prediksi.y=prediksi.kelas)
    return(hasil)
  }
}

print("Adaboost SVM Linear")
set.seed(133)
datafix1=cbind(latih[,-ncol(latih)],y=latih$Churn)
datafix1$y=ifelse(datafix1$y==1,1,-1)
datacoba=stratified.cv(datafix1,5)
konfusi=vector("list",5)
out=vector("list",5)
gmean=c()
akurasi=c()
gmean.iter=c()
akurasi.iter=c()
sens.iter=vector("list",5)
iterasi=seq(1,10,by=1)
matriks.akurasi=matrix(0,nrow=5,ncol=20)
matriks.gmean=matrix(0,nrow=5,ncol=20)
cost=c(0.01,0.1,1,10,100)
for(l in 1:length(cost)){
  cat("model svm linear pada cost",cost[l],"\n")
  for (j in 1:length(iterasi)){
    cat(j," iterasi","\n")
    for (i in 1:5){
      cat("validasi ke-",i,"\n")
      adaboost=adaboostori(X=datacoba[[1]][[i]][,-ncol(datacoba[[1]][[i]])],y=datacoba[[1]][[i]][,ncol(datacoba[[1]][[i]])],X.test=datacoba[[2]][[i]][,-ncol(datacoba[[2]][[i]])],y.test=datacoba[[2]][[i]][,ncol(datacoba[[2]][[i]])],kernel="linear",cost=cost[l],iterasi=iterasi[j])
      prediksi.boost=adaboost$prediksi.y
      prediksi.boost=as.factor(prediksi.boost)
      levels(prediksi.boost)=c("-1","1")
      
      akurasi[i]=confusionMatrix(table(prediksi.boost,datacoba[[2]][[i]][,ncol(datacoba[[2]][[i]])]))$overall[1]
      
      konfusi[[i]]=confusionMatrix(table(prediksi.boost,datacoba[[2]][[i]][,ncol(datacoba[[2]][[i]])]))$table
      
      out[[i]]=confusionMatrix(table(prediksi.boost,datacoba[[2]][[i]][,ncol(datacoba[[2]][[i]])]))$byClass[c(1,2)]
      gmean[i]=sqrt((out[[i]][1])*(out[[i]][2]))
    }
    print(gmean)
    gmean.iter[j]=mean(gmean)
    print(akurasi)
    akurasi.iter[j]=mean(akurasi)
    print(out)
    #print(konfusi)
    tab=do.call("cbind",out)
    sen=apply(tab,1,mean)
    print(sen)
    sens.iter[[j]]=sen
  }
  cat("nilai akurasi tiap iterasi","\n")
  print(akurasi.iter)
  matriks.akurasi[l,]=akurasi.iter
  cat("nilai gmean tiap iterasi","\n")
  print(gmean.iter)
  matriks.gmean[l,]=gmean.iter
  cat("nilai sensitivity dan specificity tiap iterasi","\n")
  print(sens.iter)
}
cat("hasil akurasi untuk setiap tiap cost maks 10 iterasi","\n")
print(matriks.akurasi)
cat("hasil gmean untuk setiap tiap cost maks 10 iterasi","\n")
print(matriks.gmean)

message("Adaboost SVM RBF")
print("Adaboost SVM Radial")
set.seed(663)
datafix1=cbind(latih[,-ncol(latih)],y=latih$Churn)
datafix1$y=ifelse(datafix1$y==1,1,-1)
datacoba=stratified.cv(datafix1,5)
konfusi=vector("list",5)
out=vector("list",5)
gmean=c()
akurasi=c()
gmean.iter=c()
akurasi.iter=c()
iterasi=seq(1,10,by=1)
sens.iter=vector("list",5)
cost=c(0.025,0.25,2.5,25,250)
gamma=c(0.01,0.1,1,10,100)
matriks.akur=vector("list",length(cost))
matriks.gm=vector("list",length(cost))
matriks.akurasi=matrix(0,nrow=length(gamma),ncol=length(iterasi))
matriks.gmean=matrix(0,nrow=length(gamma),ncol=length(iterasi))
for(l in 1:length(cost)){
  for (k in 1:length(gamma)){
    cat("model svm radial pada pasangan cost",cost[l],"gamma=",gamma[k],"\n")
    for (j in 1:length(iterasi)){
      cat(j,"iterasi digunakan","\n")
      for (i in 1:5){
        cat("validasi ke-",i,"\n")
        adaboost=adaboostori(X=datacoba[[1]][[i]][,-ncol(datacoba[[1]][[i]])],y=datacoba[[1]][[i]][,ncol(datacoba[[1]][[i]])],X.test=datacoba[[2]][[i]][,-ncol(datacoba[[2]][[i]])],y.test=datacoba[[2]][[i]][,ncol(datacoba[[2]][[i]])],kernel="radial",cost=cost[l],gamma=gamma[k],iterasi=iterasi[j])
        prediksi.boost=adaboost$prediksi.y
        prediksi.boost=as.factor(prediksi.boost)
        levels(prediksi.boost)=c("-1","1")
        
        akurasi[i]=confusionMatrix(table(prediksi.boost,datacoba[[2]][[i]][,ncol(datacoba[[2]][[i]])]))$overall[1]
        
        konfusi[[i]]=confusionMatrix(table(prediksi.boost,datacoba[[2]][[i]][,ncol(datacoba[[2]][[i]])]))$table
        
        out[[i]]=confusionMatrix(table(prediksi.boost,datacoba[[2]][[i]][,ncol(datacoba[[2]][[i]])]))$byClass[c(1,2)]
        gmean[i]=sqrt((out[[i]][1])*(out[[i]][2]))
      }
      print(gmean)
      gmean.iter[j]=mean(gmean)
      print(akurasi)
      akurasi.iter[j]=mean(akurasi)
      print(out)
      #print(konfusi)
      tab=do.call("cbind",out)
      sen=apply(tab,1,mean)
      print(sen)
      sens.iter[[j]]=sen
    }
    cat("nilai akurasi 20 iterasi","\n")
    print(akurasi.iter)
    matriks.akurasi[k,]=akurasi.iter
    cat("nilai gmean 20 iterasi","\n")
    print(gmean.iter)
    matriks.gmean[k,]=gmean.iter
    cat("nilai sensitivity dan specificity 20 iterasi maks","\n")
    print(sens.iter)
  }
  matriks.akur[[l]]=matriks.akurasi
  matriks.gm[[l]]=matriks.gmean
}
cat("hasil akurasi untuk setiap pasangan cost gamma maks 20 iterasi","\n")
print(matriks.akur)
cat("hasil gmean untuk setiap pasangan cost gamma maks 20 iterasi","\n")
print(matriks.gm)
