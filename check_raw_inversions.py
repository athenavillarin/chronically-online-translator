import pandas as pd
import re

# Load raw datasets
genz_raw = pd.read_csv(r'data/raw/genz_dataset.csv')
synthetic_raw = pd.read_csv(r'data/raw/synthetic_genz.csv')

print('='*80)
print('CHECKING RAW DATASET STRUCTURE')
print('='*80)
print(f'\ngenz_dataset.csv columns: {genz_raw.columns.tolist()}')
print(f'First 3 rows:')
print(genz_raw.head(3))

print(f'\n\nsynthetic_genz.csv columns: {synthetic_raw.columns.tolist()}')
print(f'First 3 rows:')
print(synthetic_raw.head(3))

# Check for inverted rows in raw data
print('\n' + '='*80)
print('CHECKING FOR INVERSIONS IN RAW DATASETS')
print('='*80)

def has_gen_z_markers(text):
    patterns = [r'\b(fr|ngl|istg|lowkey|bussin|no cap|finna|tryna|deadass|hella|mad|drained|chillin|vibes|energy|aura|af|bruh|yo|peep|lit|slaps|tea|sus|slay)\b']
    return bool(re.search('|'.join(patterns), str(text).lower()))

# Check genz_dataset
print('\nIn genz_dataset.csv (should be: normal -> gen_z):')
inverted_genz = []
for idx, row in genz_raw.iterrows():
    normal_is_genz = has_gen_z_markers(row['normal'])
    genz_is_genz = has_gen_z_markers(row['gen_z'])
    
    if normal_is_genz and not genz_is_genz:
        inverted_genz.append(idx)
        print(f'  Row {idx}: INVERTED')
        print(f'    normal: {str(row["normal"])[:60]}')
        print(f'    gen_z: {str(row["gen_z"])[:60]}')

print(f'Total inverted in genz_dataset: {len(inverted_genz)}')

# Check synthetic_genz
print('\nIn synthetic_genz.csv (should be: normal -> gen_z):')
inverted_synthetic = []
for idx, row in synthetic_raw.iterrows():
    normal_is_genz = has_gen_z_markers(row['normal'])
    genz_is_genz = has_gen_z_markers(row['gen_z'])
    
    if normal_is_genz and not genz_is_genz:
        inverted_synthetic.append(idx)
        print(f'  Row {idx}: INVERTED')
        print(f'    normal: {str(row["normal"])[:60]}')
        print(f'    gen_z: {str(row["gen_z"])[:60]}')

print(f'Total inverted in synthetic_genz: {len(inverted_synthetic)}')
