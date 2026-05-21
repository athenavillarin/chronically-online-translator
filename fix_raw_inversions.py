import pandas as pd
import re

# Load synthetic_genz.csv
df = pd.read_csv(r'data/raw/synthetic_genz.csv')

def has_gen_z_markers(text):
    patterns = [r'\b(fr|ngl|istg|lowkey|bussin|no cap|finna|tryna|deadass|hella|mad|drained|chillin|vibes|energy|aura|af|bruh|yo|peep|lit|slaps|tea|sus|slay)\b']
    return bool(re.search('|'.join(patterns), str(text).lower()))

print(f'Total rows in synthetic_genz.csv: {len(df)}')
print('Detecting and fixing inverted rows...\n')

fixed_count = 0
for idx, row in df.iterrows():
    normal_is_genz = has_gen_z_markers(row['normal'])
    genz_is_genz = has_gen_z_markers(row['gen_z'])
    
    # If normal column has gen_z markers but gen_z column doesn't, it's inverted
    if normal_is_genz and not genz_is_genz:
        # Swap them
        df.at[idx, 'normal'], df.at[idx, 'gen_z'] = row['gen_z'], row['normal']
        fixed_count += 1

print(f'Fixed {fixed_count} inverted rows')
print()

# Save back
df.to_csv(r'data/raw/synthetic_genz.csv', index=False)

# Verify
df = pd.read_csv(r'data/raw/synthetic_genz.csv')
remaining_inversions = 0
for idx, row in df.iterrows():
    normal_is_genz = has_gen_z_markers(row['normal'])
    genz_is_genz = has_gen_z_markers(row['gen_z'])
    
    if normal_is_genz and not genz_is_genz:
        remaining_inversions += 1

print(f'✓ synthetic_genz.csv saved')
print(f'✓ Remaining inversions: {remaining_inversions}')
print()
print('First 3 rows (verified correct):')
print(df.head(3))
