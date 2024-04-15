from rdkit import Chem
from rdkit.Chem.QED import properties
import pandas as pd
from rdkit.Chem.MolSurf import SlogP_VSA2,PEOE_VSA8
from rdkit.Chem import GraphDescriptors
from rdkit.Chem.Fragments import fr_Ndealkylation1, fr_piperdine
from rdkit.Chem import AllChem
from rdkit.Chem.EState.EState_VSA import VSA_EState9
import numpy as np

def get_seq_feature(seq):
    mol = Chem.MolFromFASTA(seq)
    p = properties(mol)
    return list(p)
def get_feature(seq,sym_dic):
    feat = []
    for s in seq:
        f = get_smile_feature(sym_dic[s])
        feat.append(f)
    feat = np.array(feat)
    a = feat.mean(axis=0)
    b = feat.max(axis=0)
    c = feat.min(axis=0)
    d = feat.std(axis=0)
    e = feat.sum(axis=0)
    return np.hstack([a,b,c,d,e])

def calculate_chirality(smiles):
    mol = Chem.MolFromSmiles(smiles)
    chiral_centers = Chem.FindMolChiralCenters(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    left_count = 0
    right_count = 0

    for atom, _ in chiral_centers:
        stereo = mol.GetAtomWithIdx(atom).GetChiralTag()
        if stereo == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            right_count += 1
        elif stereo == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            left_count += 1

    return left_count, right_count

def get_statistical(feat):
    feat = np.array(feat)
    a = feat.mean(axis=0)
    b = feat.max(axis=0)
    c = feat.min(axis=0)
    d = feat.std(axis=0) # distribution of charge
    e = feat.sum(axis=0)
    return np.hstack([a,b,c,d, e])

def calculate_skeleton_length(mol):
    skeleton_length = 0
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            skeleton_length += 1
    return skeleton_length
from rdkit.Chem.MolStandardize import rdMolStandardize

def get_smile_feature(smiles, use_charge = True):
    from rdkit import Chem
    from rdkit.Chem import Descriptors  
    mol = Chem.MolFromSmiles(smiles)
    p = properties(mol)
    MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS = p
    #MW = np.log(MW)
    p = [MW, ALOGP, PSA,]
    left, right = calculate_chirality(smiles)
    fingerprint= AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    
    if use_charge:
        AllChem.ComputeGasteigerCharges(mol)

        contribs = [float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(mol.GetNumAtoms())]
        feat = get_statistical(contribs)
        c_mean, c_max, c_min, c_std, c_sum = feat

    polarizability = Descriptors.MolMR(mol)
    vsa = VSA_EState9(mol)
    ndeal = fr_Ndealkylation1(mol)
    kappa1 = GraphDescriptors.Kappa1(mol)
    slogp = SlogP_VSA2(mol)
    peoe = PEOE_VSA8(mol)

    if use_charge:
        c_f = [np.log(MW), ALOGP, PSA, vsa, slogp, c_max, c_min, c_mean, c_sum, c_std, polarizability, peoe, kappa1]
        d_f = [HBA, HBD, ROTB, AROM, ndeal,left, right]
    else:
        c_f = [np.log(MW), ALOGP, PSA, vsa, polarizability]
        d_f = [HBA, HBD, ROTB, AROM]

    feat =  np.array(c_f + d_f + list(fingerprint)).astype(float)
    return feat
