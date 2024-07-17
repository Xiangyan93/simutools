#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from rdkit import Chem
from rdkit.Chem import rdmolops


def mol_filter(smiles: str, excluded_smarts: List[str], heavy_atoms: Tuple[int, int] = None) -> Optional[str]:
    """ Filter a molecule.

    Parameters
    ----------
    smiles: The SMILES of a molecule.
    excluded_smarts: A list of SMARTS representing the obsolete substructures.
    heavy_atoms: The valid heavy atom range.

    Returns
    -------
    The SMILES of the molecule if it is qualified. None otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    # Return None for heavy atoms out of range.
    if heavy_atoms is not None:
        if not heavy_atoms[0] <= mol.GetNumAtoms() <= heavy_atoms[1]:
            return None
    # Return None for wrong smiles.
    if mol is None:
        print('Ignore invalid SMILES: %s.' % smiles)
        return None
    # return None for molecules contain bad smarts.
    for smarts in excluded_smarts or []:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts),
                                          useChirality=True,
                                          maxMatches=1)
        if len(matches) != 0:
            return None
    # return canonical smiles.
    return Chem.MolToSmiles(mol)


def get_format_charge(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return rdmolops.GetFormalCharge(mol)
