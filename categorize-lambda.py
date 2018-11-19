#importing libraries
import boto3
from io import BytesIO
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from contextlib import closing
from boto3.dynamodb.conditions import Key, Attr
import numpy as np

def predict(input_doc):
    
    #Downloading the model and Tfidf vocabulary used for training from S3 bucket
    bucket = boto3.resource("s3").Bucket("pydata.heavywater")
    with BytesIO() as modelfo:
        bucket.download_fileobj("model/rf_clf.pkl", Fileobj = modelfo)
        model = joblib.load(modelfo)
        bucket.download_fileobj("model/tfidf.pkl", Fileobj = modelfo)
        tfidf = joblib.load(modelfo)

    #predicting the class of the input document
    tfidf_vectorizer=TfidfVectorizer(decode_error="replace",vocabulary = tfidf.vocabulary_)
    test_features=tfidf_vectorizer.fit_transform([input_doc]).toarray()
    prediction=model.predict(test_features)

    return prediction.tolist()[0]
            
def lambda_handler(event, context):
    postId = event["Records"][0]["Sns"]["Message"]
    print ("Post ID in DynamoDB: " + postId)
    
    #Retrieving information about the post from DynamoDB table
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table("posts")
    postItem = table.query(
        KeyConditionExpression=Key('id').eq(postId)
    )
    
    text = postItem["Items"][0]["text"]
    result = predict(text)
    
    #Updating the item in DynamoDB
    response = table.update_item(
        Key={'id':postId},
          UpdateExpression=
            "SET #statusAtt = :statusValue, #classAtt = :classValue",                   
          ExpressionAttributeValues=
            {':statusValue': 'UPDATED', ':classValue': result},
        ExpressionAttributeNames=
          {'#statusAtt': 'status', '#classAtt': 'class'},
    )
    
    return

