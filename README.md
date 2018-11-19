# HeavyWater---Project

### Aim:
Training a document classification model and deploying it to a public cloud platform(AWS) as a webservice

### Data: 
It consists of Document Label and a space delimited set of word values.

### Preprocessing:
Since the data is unbalanced, I used undersampling method to balance my dataset. As a result the dataset has come down to 229 records for each label. I tried to train my model on whole dataset but the lambda function is unable to unzip the files because of the exceeded size of the pkl files

### Architecture:
![architecture](https://user-images.githubusercontent.com/33338718/48680069-04370280-eb5d-11e8-88a8-06ebad64d635.png)

### Model:

I used Random Forest model available in the scikit-learn package. Trained the model on the undersampled document set and tested the model with various different sizes of test data.

The model predicts the test data with 78% accuracy. For larger test data sizes the model consistently provides an accurracy of 75%.

For the whole dataset, ANN ran into memory issues and even Naive Bayes, SVM and Random Forest models produced more than 560 MB pickle files which upon unzipping exceeded lambda function's acceptable limit. Hence I am trying to figure out if I can achive this by overcoming limitations on lambda. For now, I stick to use the undersampled dataset which has less number of records.

### Deployment:

The prediction function is deployed on AWS lambda using serverless. The model and required packages with dependencies for the functions are stored on AWS S3.

Steps taken for training the model and building the web service.

1. Undersampling Data
2. Creating of feature vectors using tfidf vectorizer
3. Training the Random Forest model on undersampled data
4. Saving the model and feature vocabulary as pickle files to be used in the prediction function
5. Creating the DynamoDb table to store the documents and predicted classes along with the document status.
6. Creating API on AWS API GATEWAY with GET and POST methods. POST method is integrated with Lambda function 'New Post' to store the        document into DynamoDB table and update the status as PROCESSING. GET method is integrated with Lambda function 'Get Post' to            retrieve the document, predicted class and status
7. Deploying the prediction function to AWS lamda using serverless(Deploying a function to lambda involves building required libraries      for prediction function from source in the Docker container. A docker image of Ubuntu can be downloaded [here](https://www.docker.com/products/docker-engine) .The built libraries and lamda_function need to be uploaded on S3 as a .zip file)
   The 'categorize-lambda' prediction function predicts the label and updates the status as UPDATED in DynamoDB table.
8. Hosting the website on AWS S3. Building a UI for submitting requests to the API using HTML, CSS and JavaScript files

Improvements

I didnt make use of all the data to train my model because of the size constraint on the lambda when files are unzipped. However, by utilizing all the data we can train the model more accurately and improve the prediction results.
Also by removing the stop-words and using NLP techniques like lemmatization we can refine the text input to the model for better classification.
We can try using other models like Artificial Neural Networks and fine tuning the parameters help us decide on the best model and improve the results

