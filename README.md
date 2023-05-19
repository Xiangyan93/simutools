# simutools
Tools for molecular simulation.


```commandline
conda install -c conda-forge acpype
conda install -c conda-forge packmol
pip install -e .
```

```commandline
python3 AutoMartini.py --save_dir sorafenib --res_name SOR --smiles "CNC(=O)C1=NC=CC(OC2=CC=C(NC(=O)NC3=CC(=C(Cl)C=C3)C(F)(F)F)C=C2)=C1" --action all-atom
python3 AutoMartini.py --save_dir sorafenib --res_name SOR --action cg-mapping
python3 AutoMartini.py --save_dir sorafenib-14-15 --res_name SOR --action cg-mapping --dihedral_with_force 14 15 --name sorafenib
python3 AutoMartini.py --save_dir sorafenib --res_name SOR --action bond-opt --n_iter 50
```