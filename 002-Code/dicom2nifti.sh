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
if [ -z ${OUTPUT_PATH+x} ]; then
	echo "ERROR: No output path given. You must specify the path to create the nifti with the -output_path variable."
	exit
fi


for SUBJECT in $(ls $INPUT_PATH)
do
	echo "Processing subject: $SUBJECT"

	SUBJECT_PATH="$INPUT_PATH/$SUBJECT"

	for SEQUENCE in $(ls $SUBJECT_PATH)
	do
		echo -e "\tSequence: $SEQUENCE"

		OUTPUT_SEQUENCE_PATH="$OUTPUT_PATH/$SUBJECT/$SEQUENCE"
		INPUT_SEQUENCE_PATH="$SUBJECT_PATH/$SEQUENCE"

		if [ ! -d $OUTPUT_SEQUENCE_PATH ]; then
			mkdir -p $SEQ_PATH
		fi
		if [ -z "$(ls -A $OUTPUT_SEQUENCE_PATH)" ]; then
			dcm2niix -z y -f ${SUBJECT}_${SEQUENCE} -o $OUTPUT_SEQUENCE_PATH $INPUT_SEQUENCE_PATH
		fi

	done
	echo "";

done
