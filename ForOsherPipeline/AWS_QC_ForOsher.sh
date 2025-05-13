#!/bin/bash

# Define today's date and how many days before
end_date=$(date -I)  # today's date
start_date=$(date -I -d "14 days ago")  # X days ago

# S3 bucket and destination path
s3_bucket="s3://respirai-vitals-archive"
local_dir="${base_path}/Records"  # Using base_path variable

# Loop through the date range (from X days ago to today)
current_date="$start_date"
while [[ "$current_date" < "$end_date" ]] || [[ "$current_date" == "$end_date" ]]; do
  echo "Downloading denoised and O2_Oxyfit files from $current_date..."
  
  # Sync denoised files and O2_Oxyfit CSV files for the current date from S3 to the local directory
  aws s3 cp "$s3_bucket/$current_date/" "$local_dir/$current_date/" --recursive \
      --exclude "*" --include "*denoised.csv" \
      --include "*O2.csv"
  
  # Increment the date by one day
  current_date=$(date -I -d "$current_date + 1 day")
done

echo "Download of denoised and O2_Oxyfit files from the last X days completed."

