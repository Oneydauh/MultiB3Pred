# this code is modified from torchdrug 
# https://github.com/DeepGraphLearning/torchdrug

from copy import copy
from rdkit import Chem
import torch
import warnings

bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
atom2valence = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 7}
bond2valence = [1, 2, 3, 1.5]
id2bond = {v: k for k, v in bond2id.items()}
empty_mol = Chem.MolFromSmiles("")
dummy_mol = Chem.MolFromSmiles("CC")
# orderd by perodic table
atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I","Bi"]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(7)
num_hs_vocab = range(7)
formal_charge_vocab = range(-5, 6)
chiral_tag_vocab = range(4)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))

bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))

# orderd by molecular mass
residue_vocab = ["GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "ILE", "LEU", "ASN",
                 "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"]
def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature

def bond_default(bond):
    return onehot(bond.GetBondType(), bond_type_vocab) + \
           onehot(bond.GetBondDir(), bond_dir_vocab) + \
           onehot(bond.GetStereo(), bond_stereo_vocab) + \
           [int(bond.GetIsConjugated()), int(bond.GetIsAromatic())]

def atom_default(atom):
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
           onehot(atom.GetNumRadicalElectrons(), num_radical_vocab) + \
           onehot(atom.GetHybridization(), hybridization_vocab) + \
           [atom.GetMass(),] + \
           [atom.GetIsAromatic(), atom.IsInRing()]



feature_dict = {
    'features.atom.default':atom_default,
    'features.bond.default':bond_default
}

def get_molecule_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_feature = ['default',]
    bond_feature =  ['default',]
    #mol_feature =  ['default',]

    atom_type = []
    formal_charge = []
    explicit_hs = []
    chiral_tag = []
    radical_electrons = []
    atom_map = []
    _atom_feature = []
    dummy_atom = copy(dummy_mol).GetAtomWithIdx(0)
    atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())] + [dummy_atom]

    for atom in atoms:
        atom_type.append(atom.GetAtomicNum())
        formal_charge.append(atom.GetFormalCharge())
        explicit_hs.append(atom.GetNumExplicitHs())
        chiral_tag.append(atom.GetChiralTag())
        radical_electrons.append(atom.GetNumRadicalElectrons())
        atom_map.append(atom.GetAtomMapNum())
        feature = []
        for name in atom_feature:
            func = feature_dict.get("features.atom.%s" % name)
            feature += func(atom)
        _atom_feature.append(feature)
    atom_type = torch.tensor(atom_type)[:-1]
    atom_map = torch.tensor(atom_map)[:-1]
    formal_charge = torch.tensor(formal_charge)[:-1]
    explicit_hs = torch.tensor(explicit_hs)[:-1]
    chiral_tag = torch.tensor(chiral_tag)[:-1]
    radical_electrons = torch.tensor(radical_electrons)[:-1]
    if len(atom_feature) > 0:
        _atom_feature = torch.tensor(_atom_feature)[:-1]
    else:
        _atom_feature = None

    edge_list = []
    bond_type = []
    bond_stereo = []
    stereo_atoms = []
    _bond_feature = []
    dummy_bond = copy(dummy_mol).GetBondWithIdx(0)
    bonds = [mol.GetBondWithIdx(i) for i in range(mol.GetNumBonds())] + [dummy_bond]
    for bond in bonds:
        type = str(bond.GetBondType())
        stereo = bond.GetStereo()
        if stereo:
            _atoms = [a for a in bond.GetStereoAtoms()]
        else:
            _atoms = [0, 0]
        if type not in bond2id:
            continue
        type = bond2id[type]
        h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_list += [[h, t, type], [t, h, type]]
        # always explicitly store aromatic bonds, no matter kekulize or not
        if bond.GetIsAromatic():
            type = bond2id["AROMATIC"]
        bond_type += [type, type]
        bond_stereo += [stereo, stereo]
        stereo_atoms += [_atoms, _atoms]
        feature = []
        for name in bond_feature:
            func = feature_dict.get("features.bond.%s" % name)
            feature += func(bond)
        _bond_feature += [feature, feature]
    edge_list = edge_list[:-2]
    bond_type = torch.tensor(bond_type)[:-2]
    bond_stereo = torch.tensor(bond_stereo)[:-2]
    stereo_atoms = torch.tensor(stereo_atoms)[:-2]
    if len(bond_feature) > 0:
        _bond_feature = torch.tensor(_bond_feature)[:-2]
    else:
        _bond_feature = None

    return _atom_feature, _bond_feature, edge_list