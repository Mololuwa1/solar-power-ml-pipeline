import json
import boto3
from datetime import datetime, timedelta

def lambda_handler(event, context):
    """Check for new data in S3 bucket"""
    
    s3_client = boto3.client('s3')
    bucket = event['bucket']
    prefix = event['prefix']
    
    # Check for files modified in the last 24 hours
    cutoff_time = datetime.now() - timedelta(days=1)
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        new_files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['LastModified'].replace(tzinfo=None) > cutoff_time:
                    new_files.append(obj['Key'])
        
        has_new_data = len(new_files) > 0
        
        return {
            'statusCode': 200,
            'hasNewData': has_new_data,
            'newFiles': new_files,
            'fileCount': len(new_files)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'error': str(e),
            'hasNewData': False
        }
