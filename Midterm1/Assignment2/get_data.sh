# downloads the dataset if it isn't in the work dir
if [[ ! -f "energydata_complete.csv" ]]
then
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv;
fi