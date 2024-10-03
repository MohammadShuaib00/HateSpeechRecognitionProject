import os

class AWSS3Sync:
    # Upload file to S3 bucket
    def sync_folder_to_s3(self, s3_bucket_url, filepath, filename):
        command = f"aws s3 cp {filepath}/{filename} s3://{s3_bucket_url}/"
        os.system(command)

    # Download file from S3 bucket
    def sync_folder_from_s3(self, s3_bucket_url, filename, destination):
        command = f"aws s3 cp s3://{s3_bucket_url}/{filename} {destination}/{filename}"
        os.system(command)
