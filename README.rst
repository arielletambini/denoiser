Denoiser: A nuisance regression tool for fMRI BOLD data
=======================================================

``Denoiser`` is a tool for removing sources of noise from and performing temporal filtering
of functional MRI data. It also provides visualization of the content of nuisance signals
(including *motion* information, if provided), allowing the user to get a quick sense of
data quality before and after noise removal.

Denoiser acts on 4D fMRI data (takes either a nifti file path or an already loaded nibabel
object as input). Nuisance signal removal and temporal filtering are performed on a
voxel-wise level, and a 'cleaned' 4D Nifti file (``_NR`` file)/ nibabel object are created
as outputs.

The specific noise signals to be removed are specified by the user (contained in a tsv file).
This tool should be used only **after** BOLD data are *minimally preprocessed* (for example,
after preprocessing the data using ``fmriprep``, which creates a .tsv file containing nuisance signals).

