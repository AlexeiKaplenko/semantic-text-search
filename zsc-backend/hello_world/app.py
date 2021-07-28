import json
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import sys
import logging
import traceback
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

start = time.time()

classifier = pipeline("zero-shot-classification", model="./model", device=-1) # to utilize GPU

end = time.time()
difference = end - start

logger.info(f"pipeline initialization: {difference}")

def lambda_handler(event, context):

    try:
        start = time.time()
	
        logger.info(f"event: {event}")
        logger.info(f"event['body']: {event['body']}")
	
        body = json.loads(event['body'])
        logger.info(f"body: {body}")

        candidate_labels = body['label']
        sequence = body['premise']
        
        prediction = classifier(sequence, candidate_labels, multi_class=False)
        
        output = str(prediction['scores'][0])

        print('label: {0}, probability: {1}'.format(candidate_labels, output))
        
        end = time.time()
        
        difference = end - start
        
        logger.info(f"pipeline execution: {difference}")

        return {
	    'statusCode': 200,
	    'body': json.dumps({
	        'premise': sequence,
	        'probability': output
	    })
        }
    
    except Exception as exp:
        exception_type, exception_value, exception_traceback = sys.exc_info()
        traceback_string = traceback.format_exception(exception_type, exception_value, exception_traceback)
        err_msg = json.dumps({
            "errorType": exception_type.__name__,
            "errorMessage": str(exception_value),
            "stackTrace": traceback_string
        })
        logger.error(err_msg)
    

