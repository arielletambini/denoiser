#!/usr/bin/env bash
set -x
set -e
if [ ! -e test_data ]; then
  mkdir test_data
  wget http://openneuro.outputs.s3.amazonaws.com/091b91a257f6b517790f2fb82a784c8e/3e22d7c1-c7e8-4cc7-bfdd-c89dd6681982/fmriprep/sub-01/func/sub-01_task-rhymejudgment_bold_space-MNI152NLin2009cAsym_preproc.nii.gz -O test_data/sub-01_task-rhymejudgment_bold_space-MNI152NLin2009cAsym_preproc.nii.gz
  wget http://openneuro.outputs.s3.amazonaws.com/091b91a257f6b517790f2fb82a784c8e/3e22d7c1-c7e8-4cc7-bfdd-c89dd6681982/fmriprep/sub-01/func/sub-01_task-rhymejudgment_bold_confounds.tsv -O test_data/sub-01_task-rhymejudgment_bold_confounds.tsv
fi
