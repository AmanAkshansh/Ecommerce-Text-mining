 
                #  Case Study 6 - Text Mining

#  Loading the required libraries:

library(stringr)
library(RCurl)
library(XML)
library(sqldf)
library(plyr)

#  Getting the data:

yelp<- read.csv("E:\\Analytixlabs\\Module 6 (Data science using R)\\Case Studies\\Case study 6 - Text Mining\\yelp.csv",stringsAsFactors = FALSE)
View(yelp)

anyNA(yelp)  #  No missing values in the dataset

yelp$business_id<- NULL
yelp$date<- NULL
yelp$type<- NULL

length(yelp$review_id[duplicated(yelp$review_id)]) # All Reviews are unique
yelp$user_id<- NULL
View(yelp)

# Getting the list of all the positive and negative words

pos.words = scan('E:\\Analytixlabs\\Module 6 (Data science using R)\\Class files\\Class 14\\Additional Example\\positive_words.txt',
           what='character', comment.char=';')
print(pos.words)

neg.words = scan('E:\\Analytixlabs\\Module 6 (Data science using R)\\Class files\\Class 14\\Additional Example\\negative_words.txt',
           what='character', comment.char=';')
print(neg.words)


# User defined function for calculating the sentiment score for each Review:

score.sentiment = function(sentences, pos.words, neg.words)
{
  
  scores = lapply(sentences, function(sentence, pos.words, neg.words) {
    
    # clean up sentences with R's regex-driven global substitute, gsub():
    sentence = gsub('[[:punct:]]', '', sentence)
    sentence = gsub('[[:cntrl:]]', '', sentence)
    sentence = gsub('\\d+', '', sentence)
    # and convert to lower case:
    sentence = tolower(sentence)
    
    # split into words. str_split is in the stringr package
    word.list = str_split(sentence, '\\s+')
    # sometimes a list() is one level of hierarchy too much
    words = unlist(word.list)
    
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    
    # match() returns the position of the matched term or NA if no matching
    # is found
    # we just want a TRUE/FALSE:
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
    score = sum(pos.matches) - sum(neg.matches)
    
    return(score)
  }, pos.words, neg.words )
  
  scores.df = data.frame(score=scores, text=sentences)
  return(scores.df)
}


#  Looping through each Review to find their sentiment scores
#  we also Normalize the score


for(i in 1:nrow(yelp)){
  yelp$scores[i]<- ((score.sentiment(yelp$text[i],pos.words,neg.words))/10)*5
}

View(yelp)
str(yelp)
yelp$scores<- as.numeric(yelp$scores)

yelp$review_type[yelp$scores>0]<- "Positive"
yelp$review_type[yelp$scores==0]<- "Neutral"
yelp$review_type[yelp$scores<0]<- "Negative"

hist(yelp$scores)
# The businesses provided by yelp.com are mostly reviewed Positive by the 
# users. We can also check this by the Stars given by the users

prop.table(table(yelp$stars))
# Around 70% of the reviewers gave Star rating more than 3


#########++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##########

#  Now we will start with the Model building exercise

yelp$stars<- factor(yelp$stars)
str(yelp)


#  Splitting the data into training and testing datasets:
set.seed(345)
ind<- sample(2,nrow(yelp),replace = TRUE,prob = c(0.7,0.3))
Training<- yelp[ind==1,]
Testing<- yelp[ind==2,]

View(Training)
str(Training)

prop.table(table(Training$stars))
prop.table(table(Testing$stars))
# Proportion of Stars in Training and Testing datasets is similar to the 
# original Yelp dataset.

# Data Preparation:
names(Training)

Training<- subset(Training,select = -c(review_id,text,review_type))
View(Training)

#  Model Building:

       # (1) We will use KNN first for creating the model

library(caret)
cntrl<- trainControl(method = "repeatedcv",number = 10,repeats = 3)

set.seed(555)
fit<- train(stars ~ ., data = Training,
            method="knn",
            tuneGrid=expand.grid(k=1:60),
            trControl=cntrl,
            preProcess= c("center","scale"))

print(fit)
plot(fit)

#  Prediction on Test dataset:

pred_stars<- predict(fit,newdata = Testing)
head(pred_stars)

confusionMatrix(pred_stars,Testing$stars)


#  The model can be accepted because Accuracy > No Information Rate.
# But still Accuracy is very poor and there are large no. of misclassifications
#  in our prediction.


        # (2) Now will create a model using Random Forest

library(randomForest)
names(Training)

set.seed(111)
fit2<- randomForest(stars~ .,data = Training)
print(fit2)
plot(fit2)

pred2<- predict(fit2,newdata = Testing)
confusionMatrix(pred2,Testing$stars)

# Fine tuning the RF model
tuneRF(Training[,-1],Training[,1],
       ntreeTry = 300,
       plot = TRUE,
       stepFactor = 0.5,
       trace = TRUE,
       improve = 0.05)

#  Modified RF model:
set.seed(222)
modified_fit2<- randomForest(stars ~ .,data = Training,
                             ntree=300,
                             mtry=2,
                             Importance=TRUE,
                             Proximity=TRUE)
print(modified_fit2)

pred3<- predict(modified_fit2,newdata = Testing)
confusionMatrix(pred3,Testing$stars)

# Accuracy has been slightly improved than the previous RF model but still
# KNN gave us better accuracy

       # (3) We will check this time around with Naive Bayes

library(naivebayes)
names(Training)

set.seed(666)
fit3<- naive_bayes(stars~.,data = Training,
                   usekernel = TRUE,
                   laplace = 1)

print(fit3)
plot(fit3)

pred4<- predict(fit3,newdata = Testing)
confusionMatrix(pred4,Testing$stars)

#########++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##########

#  Since KNN model gave us the best accuracy so we'll go with that model

confusionMatrix(pred_stars,Testing$stars)
# The Problem lies in Class imbalance.
# We are getting less accuracy because we have huge class imbalance in 
# target variable. Almost 70% of the data have 4 or 5 stars in both Training
# and Testing Datasets.

final.df<- cbind(Testing,Predicted_Stars=pred_stars)
View(final.df)
prop.table(table(final.df$Predicted_Stars))
prop.table(table(final.df$stars))

write.csv(final.df,"Predicted Stars.csv")


###############+++++++++++++++++++++++++++++++++++++++++++++++++###############
#                             END                                             #   #
##############++++++++++++++++++++++++++++++++++++++++++++++++++###############
































