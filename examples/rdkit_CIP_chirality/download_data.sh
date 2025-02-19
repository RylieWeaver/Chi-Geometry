#!/usr/bin/env bash

# Download the data files using curl
BASEDIR="$(dirname "$0")"
DATADIR="datasets"
mkdir -p "${BASEDIR}/${DATADIR}"

echo "Downloading train data..."
curl -L "https://figshare.com/ndownloader/files/30975694?private_link=e23be65a884ce7fc8543" \
     -o "${BASEDIR}/${DATADIR}/train_raw.pkl"

echo "Downloading validation data..."
curl -L "https://figshare.com/ndownloader/files/30975703?private_link=e23be65a884ce7fc8543" \
     -o "${BASEDIR}/${DATADIR}/val_raw.pkl"

echo "Downloading test data..."
curl -L "https://figshare.com/ndownloader/files/30975679?private_link=e23be65a884ce7fc8543" \
     -o "${BASEDIR}/${DATADIR}/test_raw.pkl"

echo "All done! Files are in the '${DATADIR}' directory under ${BASEDIR}."
