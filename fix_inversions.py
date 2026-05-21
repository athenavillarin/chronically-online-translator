import pandas as pd
import re

def is_gen_z_slang(text):
    """Detect if text is likely gen Z slang based on markers."""
    if not isinstance(text, str):
        return False
    
    # Gen Z slang markers
    gen_z_patterns = [
        r'\b(fr|fr fr|ngl|istg|lowkey|highkey|bussin|slaps|no cap|cap|finna|tryna|deadass|dead|salty|bet|vibe|vibes|energy|aura|W\b|L\b)\b',
        r'\b(ong|omg|smol|sus|slay|periodt|tea|lit|iconic|dank|cheugy|main|bestie|fam|bro)\b',
        r'\b(drained|hella|mad|hyped|hype|chillin|chilling|ghosted|glow up|caught)\b',
        r'\b(iykyk|oop|n\'t|gotta|wanna|gonna|ya|u\'re|ur|abt|rly|tho|lowk|tbh|peep|dank|af|beta)\b'
    ]
    
    text_lower = text.lower()
    pattern = '|'.join(gen_z_patterns)
    
    if re.search(pattern, text_lower):
        return True
    
    return False

def fix_inversions(file_path):
    """Fix rows where source and target are inverted."""
    df = pd.read_csv(file_path)
    
    fixed_count = 0
    for idx, row in df.iterrows():
        source = row['source_text']
        target = row['target_text']
        
        source_is_gen_z = is_gen_z_slang(source)
        target_is_gen_z = is_gen_z_slang(target)
        
        # If source is gen_z but target isn't, and target appears more formal, swap them
        if source_is_gen_z and not target_is_gen_z:
            df.at[idx, 'source_text'] = target
            df.at[idx, 'target_text'] = source
            fixed_count += 1
            print(f"Row {idx}: Fixed inversion")
            print(f"  Source (standard): {source[:60]}")
            print(f"  Target (gen_z): {target[:60]}")
            print()
    
    return df, fixed_count

# Fix train.csv
print("=" * 80)
print("Fixing train.csv...")
print("=" * 80)
train_path = r'c:\Users\meagie\Desktop\chronically-online-translator\data\processed\train.csv'
train_df, train_fixed = fix_inversions(train_path)
train_df.to_csv(train_path, index=False)
print(f"✓ Fixed {train_fixed} inverted rows in train.csv\n")

# Fix test.csv
print("=" * 80)
print("Fixing test.csv...")
print("=" * 80)
test_path = r'c:\Users\meagie\Desktop\chronically-online-translator\data\processed\test.csv'
test_df, test_fixed = fix_inversions(test_path)
test_df.to_csv(test_path, index=False)
print(f"✓ Fixed {test_fixed} inverted rows in test.csv\n")

print("=" * 80)
print(f"Total rows fixed: {train_fixed + test_fixed}")
print("=" * 80)

# Verification
print("\nVerification - train.csv first 3 rows:")
print(pd.read_csv(train_path).head(3))
print("\nVerification - test.csv first 3 rows:")
print(pd.read_csv(test_path).head(3))
