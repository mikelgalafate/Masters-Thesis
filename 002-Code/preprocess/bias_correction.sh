#!/bin/bash

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY=${KEY^^}
   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "${KEY[@]:1}=$VALUE"

done

if [ -z ${INPUT_PATH+x} ]; then
	echo "ERROR: No input path given. You must specify the path to the data to be converted to nifti with the -input_path variable."
	exit
fi
if [ -z ${SEQUENCE+x} ]; then
	echo "ERROR: No sequence name given. You must specify the name of the sequence. Note that the name must be consistent with your database."
	exit
fi

cd $INPUT_PATH

for SUBJECT in $(ls -R sub-*); do
	if  [[ $SUBJECT =~ $SEQUENCE:$ ]]; then
		SUBJECT_PATH="$INPUT_PATH/${SUBJECT::-1}"
		SUBJECT_OUTPUT_PATH="${INPUT_PATH}_BC/${SUBJECT::-1}"
		echo -e "New subject to be processed found: $SUBJECT_PATH"
		continue
	elif [[ $SUBJECT =~ ([A-Za-z0-9._%+-]+)?$SEQUENCE([A-Za-z0-9._%+-]+)?.nii.gz$ ]]; then
		FILE_NAME=${SUBJECT::-7}
		echo -e "\tFound $FILE_NAME at $SUBJECT_PATH"
	else
		continue
	fi

	echo -e "\tProcessing volume $FILE_NAME"
	echo -e "\tOutput volume path: $SUBJECT_OUTPUT_PATH/$SUBJECT\n"
	if [ ! -d $SUBJECT_OUTPUT_PATH ]; then
		mkdir -p $SUBJECT_OUTPUT_PATH
	fi
	echo -e "\tVolume $FILE_NAME submitted for bias correction on `date`"

	Slicer --launch N4ITKBiasFieldCorrection --convergencethreshold 0.00001 --iterations 500,400,300 $SUBJECT_PATH/$SUBJECT $SUBJECT_OUTPUT_PATH/$SUBJECT > /dev/null

done