#!/bin/bash

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY=${KEY^^}
   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "${KEY[@]:1}=$VALUE"

done

for SUBJECT in $(ls $INPUT_PATH)
do
	echo "Processing subject: $SUBJECT"

	SUBJECT_PATH="$INPUT_PATH/$SUBJECT"

	for SEQUENCE in $(ls $SUBJECT_PATH)
	do
		echo -e "\tSequence: $SEQUENCE"

		OUTPUT_SEQUENCE_PATH="$OUTPUT_PATH/$SUBJECT/$SEQUENCE"
		INPUT_SEQUENCE_PATH="$SUBJECT_PATH/$SEQUENCE"

		if [ ! -d $SEQ_PATH ]; then
			mkdir -p $SEQ_PATH
		fi
		if [ -z "$(ls -A $SEQ_PATH)" ]; then
			echo -e "dcm2niix -z y -f ${SUBJECT}_${SEQUENCE} -o $OUTPUT_SEQUENCE_PATH $INPUT_SEQUENCE_PATH"
		fi

	done
	echo "";

done
