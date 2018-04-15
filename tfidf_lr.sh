#!/bin/bash
python module/generate_tfidf.py > log/tfidf_lr.log
python module/train_lr.py >> log/tfidf_lr.log
python module/evaluate.py >> log/tfidf_lr.log
echo 'All scripts completed' >> log/tfidf_lr.log
