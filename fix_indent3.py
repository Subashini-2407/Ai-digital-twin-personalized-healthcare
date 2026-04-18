#!/usr/bin/env python3
"""Fix remaining indentation in the Risk Analysis section of ultimate_enhanced2.py"""

with open('src/app/ultimate_enhanced2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "col_r1, col_r2, col_r3 = st.columns([1, 1, 1])" 
# Everything from there until "# ==================== LIFESTYLE OPTIMIZER ====================" 
# needs to be indented by 4 more spaces

in_section = False
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if 'col_r1, col_r2, col_r3 = st.columns([1, 1, 1])' in line and not in_section:
        in_section = True
        start_idx = i
    elif in_section and '# ==================== LIFESTYLE OPTIMIZER ====================' in line:
        end_idx = i
        break

if start_idx is not None and end_idx is not None:
    # Add 4 spaces to all lines in this range
    for i in range(start_idx, end_idx):
        if lines[i].strip():  # Non-empty line
            if lines[i].startswith('        '):  # Currently 8 spaces
                lines[i] = '            ' + lines[i][8:]  # Replace with 12 spaces
    
    with open('src/app/ultimate_enhanced2.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'Fixed indentation from line {start_idx+1} to {end_idx}')
else:
    print(f'Could not find section (start={start_idx}, end={end_idx})')
