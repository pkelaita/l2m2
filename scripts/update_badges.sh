#!/bin/bash

# For cache busting
timestamp=$(date +%s)
README_PATH="../README.md"

sed -i "" "s|timestamp=[0-9]*|timestamp=$timestamp|g" $README_PATH
echo "New timestamp: $timestamp"
