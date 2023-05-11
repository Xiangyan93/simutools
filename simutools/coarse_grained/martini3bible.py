#!/usr/bin/env python
# -*- coding: utf-8 -*-


sugar_beadtype = {
    'P4r': 'SP4r',
    'TN4a': 'TN4ar',
    'SP1': 'SN6',
    'N4a': 'N4a',
}

# SMARTS with hydrogens in RDKit.
martini3bible = {
    # carbonhydrates
    'CS(=O)(=O)C': 'P6',
    # 4-atom substructure
    'C(F)(F)F': 'SX4e',
    '[N;1H]C(=O)[N;1H]': 'P3',
    '[N;0H]C(=O)[N;0H]': 'P3a',
    'CS(=O)C': 'P6',
    # 3-atom substructure
    'C(=O)[N;1H]': 'P3',
    'C(=O)[N;0H]': 'P3a',
    '[C;D3;H0](=O)[O;D2;H0]': 'N4a',  # ester
    'S(=O)(=O)': 'P6',
    'C(=O)[O;H1]': 'P2',  # acetic acid
    # 2-atom substructure
    'S(=O)': 'P6',
    '[C;H1](=O)': 'N6a',
    # 1-atom functional group
    '[O;D1;H0]': 'N5a',  # ketone
    '[O;D2;H0]': 'N4a',  # ether
    '[O;D2;H1]': 'P1',  # hydroxyl
    '[N;D1]': 'N6a',
    '[N;D2]': 'N4',
    '[N;D3]': 'N3a',
    '[S]': 'C6',
    '[F]': 'X4e',
    '[Cl]': 'X3',
    '[Br]': 'X2',
}
