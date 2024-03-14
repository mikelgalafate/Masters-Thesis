#!/bin/bash

# Read variables
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY=${KEY^^}
   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "${KEY[@]:1}=$VALUE"

done

if [ -z ${INPUT_PATH+x} ]; then
	echo "ERROR: No input path given. You must specify the path to the data with the -input_path variable."
	exit
fi
if [ -z ${SEQUENCE+x} ]; then
	echo "ERROR: No sequence name given. You must specify the name of the sequence with the -sequence variable. Note that the name must be consistent with your database."
	exit
fi


echo -e "Copying data.."
# Remove output directory if already exists
if [ -d ${INPUT_PATH}_BC ]; then
	rm -rf ${INPUT_PATH}_BC
fi
# Copy the dataset to the output directory
cp -r $INPUT_PATH ${INPUT_PATH}_BC

echo -e "Starting!"

cd ${INPUT_PATH}_BC || exit

for SUBJECT in $(ls -R sub-*); do
	# Check the type of file
	if  [[ $SUBJECT =~ $SEQUENCE:$ ]]; then # The file is a directory containing the desired sequence
		# Save the path to the sequence directory
		SUBJECT_PATH="$INPUT_PATH/${SUBJECT::-1}"
		SUBJECT_OUTPUT_PATH="${INPUT_PATH}_BC/${SUBJECT::-1}"
		echo -e "New subject to be processed found: $SUBJECT_PATH"
		continue
	elif [[ $SUBJECT =~ ([A-Za-z0-9._%+-]+)?$SEQUENCE([A-Za-z0-9._%+-]+)?.nii.gz$ ]]; then # The file is a nifti file of the desired sequence
		# Process the file
		echo -e "\tFound ${SUBJECT::-7} at $SUBJECT_PATH"
		echo -e "\tProcessing volume ${SUBJECT::-7}"
		echo -e "\tOutput volume path: $SUBJECT_OUTPUT_PATH/$SUBJECT\n"
		if [ ! -d $SUBJECT_OUTPUT_PATH ]; then
			mkdir -p $SUBJECT_OUTPUT_PATH
		fi
		echo -e "\tVolume ${SUBJECT::-7} submitted for bias correction on `date`"
		Slicer --launch N4ITKBiasFieldCorrection --convergencethreshold 0.00001 --iterations 500,400,300 $SUBJECT_PATH/$SUBJECT $SUBJECT_OUTPUT_PATH/$SUBJECT > /dev/null
	fi
done
