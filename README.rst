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

.. image:: https://zenodo.org/badge/4033784/arielletambini/denoiser.svg
   :target: https://zenodo.org/badge/latestdoi/4033784/arielletambini/denoiser



For instructions on how to run, type: python run_denoise.py -h::

   usage: run_denoise.py [-h] [--col_names COL_NAMES [COL_NAMES ...]] [--hp_filter HP_FILTER] [--lp_filter LP_FILTER] [--out_figure_path OUT_FIGURE_PATH] img_file tsv_file out_path

   Function for performing nuisance regression. Saves resulting output nifti file, information about nuisance
   regressors and motion (html report), and outputs nibabel object containing clean data

   positional arguments:
     img_file              4d nifti img: file path or nibabel object loaded into memory
     tsv_file              tsv file containing nuisance regressors to be removed
     out_path              output directory for saving new data file

   optional arguments:
     -h, --help            show this help message and exit
     --col_names COL_NAMES [COL_NAMES ...]
                           which columns of TSV file to include as nuisance regressors. defaults to ALL columns.
     --hp_filter HP_FILTER
                           frequency cut-off for high pass filter (removing low frequencies). Recommend .009 Hz
     --lp_filter LP_FILTER
                           frequency cut-off for low pass filter (removing high frequencies). Recommend .1 Hz for non-task data
     --out_figure_path OUT_FIGURE_PATH
                           output directory for saving figures. Defaults to location of out_path + _figures


If you run into troubles using the tool or have any questions please post them `here <https://neurostars.org/tag/denoiser>`_.
