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
	echo "ERROR: No input path given. You must specify the path to the data for registration."
	exit
fi
if [ -z ${FIXED+x} ]; then
	echo "ERROR: No fixed volume. You must specify the name of the sequence to be used as fixed volume. Note that the name must be consistent with your database."
	exit
fi
if [ -z ${MOVING+x} ]; then
	echo "ERROR: No moving volume. You must specify the name of the sequence to be used as moving volume. Note that the name must be consistent with your database."
	exit
fi

TMP_DIR=$INPUT_PATH/$MOVING/tmp
MASK_PATH=$TMP_DIR/${MOVING}_mask.nii.gz
MASKED_VOLUME=$TMP_DIR/${MOVING}_masked.nii.gz
SUBJECT=$(echo -e "$INPUT_PATH" | awk -F"/" '{print $NF}')
echo -e "Processing subj: $SUBJECT"

if [ ! -d $TMP_DIR ]; then
	mkdir $TMP_DIR
fi

Slicer --launch BRAINSROIAuto --inputVolume $INPUT_PATH/$MOVING/${SUBJECT}_$MOVING.nii.gz --outputROIMaskVolume $MASK_PATH --otsuPercentileThreshold 0.01 --thresholdCorrectionFactor 1 --closingSize 9 --ROIAutoDilateSize 0 --outputVolumePixelType short --numberOfThreads -1
Slicer --launch MaskScalarVolume --label 1 --replace -1024 $INPUT_PATH/$MOVING/${SUBJECT}_$MOVING.nii.gz $MASK_PATH $MASKED_VOLUME
Slicer --launch /opt/Slicer-5.4.0-linux-amd64/slicer.org/Extensions-31938/SlicerElastix/lib/Slicer-5.4/elastix -m $MASKED_VOLUME -f $INPUT_PATH/$FIXED/${SUBJECT}_$FIXED.nii.gz -p /opt/Slicer-5.4.0-linux-amd64/slicer.org/Extensions-31938/SlicerElastix/lib/Slicer-5.4/qt-scripted-modules/Resources/RegistrationParameters/Parameters_Rigid_CT.txt -p /opt/Slicer-5.4.0-linux-amd64/slicer.org/Extensions-31938/SlicerElastix/lib/Slicer-5.4/qt-scripted-modules/Resources/RegistrationParameters/Parameters_BSpline_CT.txt -out $TMP_DIR
if [ ! -d $INPUT_PATH/${MOVING}_reg ]; then
	mkdir $INPUT_PATH/${MOVING}_reg
fi
python3 -c "import SimpleITK as sitk; import os; img = sitk.ReadImage('$TMP_DIR/result.1.mhd'); sitk.WriteImage(img, '$INPUT_PATH/${MOVING}_reg/${SUBJECT}_${MOVING}_reg.nii.gz')" && rm -rf $TMP_DIR