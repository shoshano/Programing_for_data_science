import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Fragments, Lipinski, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("COMPLETE SMILES PREPROCESSING PIPELINE")
print("="*70)

# ============ STEP 1: LOAD AND CLEAN DATA ============
print("\nSTEP 1: Load and Clean Data")
print("-" * 70)

# Load your data
data = pd.read_csv('.\\Assignment4\\datasets\\training_smiles.csv')  # Change to your file path
print(f"Original dataset shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")

# Remove duplicates and conflicting labels
duplicates = data[data.duplicated(subset="SMILES", keep=False)].sort_values(by="SMILES")
conflicting_smiles = duplicates.groupby('SMILES').filter(lambda x: x['ACTIVE'].nunique() > 1)

data_clean = data[~data['SMILES'].isin(conflicting_smiles['SMILES'])]
data_clean = data_clean.drop_duplicates(subset='SMILES', keep='first')

print(f"After removing duplicates/conflicts: {data_clean.shape}")
print(f"Removed {len(data) - len(data_clean)} rows")

# ============ STEP 2: CONVERT SMILES TO MOL OBJECTS ============
print("\n" + "="*70)
print("STEP 2: Convert SMILES to RDKit Molecules")
print("-" * 70)

def smiles_to_mol(smiles):
    """Convert SMILES string to RDKit Mol object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

print("Converting SMILES to molecules...")
data_clean['mol'] = data_clean['SMILES'].apply(smiles_to_mol)

# Filter out invalid molecules
valid_data = data_clean[data_clean['mol'].notnull()].copy()
print(f"Valid molecules: {len(valid_data)} out of {len(data_clean)}")
print(f"Invalid SMILES: {len(data_clean) - len(valid_data)}")

# ============ STEP 3: EXTRACT BASIC DESCRIPTORS ============
print("\n" + "="*70)
print("STEP 3: Extract Basic Molecular Descriptors")
print("-" * 70)

def compute_basic_descriptors(mol):
    """Extract 8 basic descriptors"""
    if mol is None:
        return [np.nan] * 8
    
    return [
        rdMolDescriptors.CalcExactMolWt(mol),      # Exact Molecular Weight
        Descriptors.MolLogP(mol),                   # LogP (lipophilicity)
        rdMolDescriptors.CalcTPSA(mol),             # Topological Polar Surface Area
        rdMolDescriptors.CalcNumRotatableBonds(mol),# Rotatable bonds
        rdMolDescriptors.CalcNumHBD(mol),           # H-Bond Donors
        rdMolDescriptors.CalcNumHBA(mol),           # H-Bond Acceptors
        Lipinski.HeavyAtomCount(mol),               # Heavy Atom Count
        Fragments.fr_Al_COO(mol),                   # Carboxylic Acid groups
    ]

descriptor_names = [
    'ExactMolWt', 'MolLogP', 'TPSA', 'NumRotBonds',
    'NumHBD', 'NumHBA', 'HeavyAtomCount', 'fr_Al_COO'
]

print("Computing basic descriptors...")
descriptors_list = []
for mol in tqdm(valid_data['mol'], desc="Descriptors"):
    descriptors_list.append(compute_basic_descriptors(mol))

descriptors_df = pd.DataFrame(descriptors_list, columns=descriptor_names)
print(f"Extracted {len(descriptor_names)} basic descriptors")
print(f"Descriptor columns: {descriptor_names}")

# ============ STEP 4: EXTRACT EXTENDED DESCRIPTORS ============
print("\n" + "="*70)
print("STEP 4: Extract Extended Descriptors")
print("-" * 70)

def compute_extended_descriptors(mol):
    """Extract 4 additional descriptors"""
    if mol is None:
        return [np.nan] * 4
    
    return [
        Descriptors.MolWt(mol),                     # Molecular Weight
        Descriptors.FractionCSP3(mol),              # Fraction sp3 carbons
        rdMolDescriptors.CalcNumHeteroatoms(mol),   # Heteroatoms
        Descriptors.NumAromaticRings(mol),          # Aromatic rings
    ]

extended_names = ['MolWt', 'FractionCSP3', 'NumHeteroatoms', 'NumAromaticRings']

print("Computing extended descriptors...")
extended_list = []
for mol in tqdm(valid_data['mol'], desc="Extended"):
    extended_list.append(compute_extended_descriptors(mol))

extended_df = pd.DataFrame(extended_list, columns=extended_names)
print(f"Extracted {len(extended_names)} extended descriptors")

# ============ STEP 5: EXTRACT FRAGMENT FEATURES ============
print("\n" + "="*70)
print("STEP 5: Extract Fragment Features")
print("-" * 70)

def compute_fragments(mol):
    """Extract 6 fragment features"""
    if mol is None:
        return [np.nan] * 6
    
    return [
        Fragments.fr_Al_COO(mol),                   # Carboxylic acid
        Fragments.fr_Ar_N(mol),                     # Aromatic nitrogen
        Fragments.fr_Ar_OH(mol),                    # Aromatic OH
        Fragments.fr_COO(mol),                      # Carboxyl
        Fragments.fr_Al_OH(mol),                    # Aliphatic OH
        Fragments.fr_ketone(mol),                   # Ketone
    ]

fragment_names = ['fr_COO', 'fr_Ar_N', 'fr_Ar_OH', 'fr_COOH', 'fr_Al_OH', 'fr_ketone']

print("Computing fragment features...")
fragments_list = []
for mol in tqdm(valid_data['mol'], desc="Fragments"):
    fragments_list.append(compute_fragments(mol))

fragments_df = pd.DataFrame(fragments_list, columns=fragment_names)
print(f"Extracted {len(fragment_names)} fragment features")

# ============ STEP 6: EXTRACT MORGAN FINGERPRINTS ============
print("\n" + "="*70)
print("STEP 6: Extract Morgan Fingerprints (Radius=2, 1024 bits)")
print("-" * 70)

def compute_morgan_fp(mol, radius=2, nBits=1024):
    """Extract Morgan fingerprint"""
    if mol is None:
        return [0] * nBits
    
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = fpgen.GetFingerprint(mol)
    return np.array(fp, dtype=int)

print("Computing Morgan fingerprints...")
morgan_list = []
for mol in tqdm(valid_data['mol'], desc="Morgan FP"):
    morgan_list.append(compute_morgan_fp(mol, radius=2, nBits=1024))

morgan_df = pd.DataFrame(morgan_list, columns=[f'Morgan_fp_{i}' for i in range(1024)])
print(f"Extracted 1024-bit Morgan fingerprints")

# ============ STEP 7: COMBINE ALL FEATURES ============
print("\n" + "="*70)
print("STEP 7: Combine All Features")
print("-" * 70)

# Reset indices to ensure alignment
valid_data = valid_data.reset_index(drop=True)
descriptors_df = descriptors_df.reset_index(drop=True)
extended_df = extended_df.reset_index(drop=True)
fragments_df = fragments_df.reset_index(drop=True)
morgan_df = morgan_df.reset_index(drop=True)

# Combine all features
X_combined = pd.concat([
    descriptors_df,
    extended_df,
    fragments_df,
    morgan_df
], axis=1)

print(f"\nTotal features: {X_combined.shape[1]}")
print(f"  - Basic descriptors: {len(descriptor_names)}")
print(f"  - Extended descriptors: {len(extended_names)}")
print(f"  - Fragments: {len(fragment_names)}")
print(f"  - Morgan fingerprints: 1024")

# Add target variable
y = valid_data['ACTIVE'].values
X_combined['ACTIVE'] = y

print(f"\nFinal dataset shape: {X_combined.shape}")
print(f"Class distribution:")
print(f"  Class 0 (Inactive): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"  Class 1 (Active): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")

# ============ STEP 8: STANDARDIZE DESCRIPTORS ============
print("\n" + "="*70)
print("STEP 8: Standardize Descriptors")
print("-" * 70)

# Standardize only the descriptor columns (not fingerprints)
descriptor_cols = descriptor_names + extended_names + fragment_names
scaler = StandardScaler()
X_combined[descriptor_cols] = scaler.fit_transform(X_combined[descriptor_cols])

print(f"Standardized {len(descriptor_cols)} descriptor columns")
print(f"Fingerprints remain as binary features")

# ============ STEP 9: SAVE PREPROCESSED DATA ============
print("\n" + "="*70)
print("STEP 9: Save Preprocessed Data")
print("-" * 70)

# Save to CSV
output_file = 'preprocessed_smiles_data.csv'
X_combined.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# Display sample of data
print(f"\nSample of first 3 rows and first 10 features:")
print(X_combined.iloc[:3, :10])

print("\n" + "="*70)
print("PREPROCESSING COMPLETE!")
print("="*70)
print(f"\nYou can now use this data for training:")
print(f"  preprocessed_data = pd.read_csv('{output_file}')")
print(f"  X = preprocessed_data.drop('ACTIVE', axis=1)")
print(f"  y = preprocessed_data['ACTIVE']")