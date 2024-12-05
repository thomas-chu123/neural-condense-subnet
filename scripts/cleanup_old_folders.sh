#!/bin/bash

# Define the parent directory where you want to check the subfolders.
PARENT_DIR="/workspace/minio/condense-miner/"

# Find subdirectories in PARENT_DIR that are older than 60 minutes and remove them.
find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d -mmin +60 -exec rm -rf {} \;
echo "cleaning on " | date >> /workspace/clean.log


# Crontabe -e
# */60 * * * * /root/cleanup_old_folders.sh