# PanDDA 2

## Installation

It is reccomended that you install PanDDA 2 in it's own python 3.9 anaconda enviroment. This can be achieved by installing anaconda and then:

```bash
conda create -n pandda2 python=3.9
conda activate pandda2
conda install -c conda-forge -y fire numpy scipy joblib scikit-learn umap-learn bokeh dask dask-jobqueue hdbscan matplotlib
conda install -c conda-forge -y seaborn
conda install -c conda-forge -y rdkit
git clone https://github.com/ConorFWild/pandda_2_gemmi.git
cd /path/to/cloned/repository
pip install .
cd _gemmi
pip install .

```

Installing PanDDA 2 this way will add various scripts to your path, but only while you are in this anaconda enviroment.



## Running

Once you have installed PanDDA 2 in a conda enviroment, running it is as simple as:

```bash
python /path/to/analyse.py <data directories> <output directory> --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz" --structure_factors='("2FOFCWT","PH2FOFCWT")' <options>

```

### Running with autobuilding
PanDDA 2 supports the autobuilding of events and ranking them by autobuildability. All one needs to do is ensure that BUSTER is setup in their path (and hence ana_pdbmaps and rhofit) and run PanDDA 2 with the autobuild flag on.

An example:
```bash
python /path/to/analyse.py <data dirs> <output dirs> --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz" --structure_factors='("2FOFCWT","PH2FOFCWT")' --autobuild=True <options>

```


### Running with distributed computing at Diamond

It is strongly reccomended that if you are qsub'ing a script that will run PanDDA 2 you set up your enviroment on the head node (by activating the anaconda enviroment in which PanDDA 2 is installed) and use the "-V" option on qsub to copy your current enviroment to the job.

An example of how to run with distributed computing at Diamond Light Source is as follows:
```bash
# Ensuring availability of Global Phasing code for autobuilding and phenix for building cifs
module load phenix
module load buster

# Put the following in the file submit.sh
python /path/to/analyse.py <data dirs> <output dirs> --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz" --structure_factors='("2FOFCWT","PH2FOFCWT")' --global_processing="distributed" <options>

# Submitting
chmod 777 submit.sh
qsub -V -o submit.o -e submit.e -q medium.q -pe smp 20 -l m_mem_free=15G submit.sh

```


### Running with distributed computing from condor
```bash
python ./pandda_gemmi/analyse.py /data/share-2/conor/pandda/data/pandda_inputs/BRD1 /data/share-2/conor/pandda/output/pandda_2_BRD1 --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz" --structure_factors='("FWT","PHWT")' --autobuild=True --global_processing="distributed" --distributed_scheduler="HTCONDOR" --local_cpus=20

```

