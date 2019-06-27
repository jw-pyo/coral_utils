export PYTHONPATH=/home/mendel/coral_utils
rm ../models/labels.txt
python3 detect.py --label-make True --knn True

