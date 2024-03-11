#!/bin/bash

path=$1
output_path=$2
#"/home/mikelgalafate/NAS/MATERIA_OSCURA/TFMs/2022_MikelGalafate/001-Data/CERMEP-IDB-MRXFDG_Database/NII_with_FDG_NAC/sourcedata"

for subject in $(ls $path)
do
	echo "Processing subject: $subject";
	for seq in $(ls $path/$subject)
	do
		echo -e "\tSequence: $seq"

		seq_path="$output_path/$subject/$seq"
		if [ ! -d $seq_path ]; then
			mkdir -p $seq_path
		fi
		if [ -z "$(ls -A $seq_path)" ]; then
			dcm2niix -z y -f ${subject}_${seq} -o $seq_path $path/$subject/$seq
		fi

	done
	echo "";

done
