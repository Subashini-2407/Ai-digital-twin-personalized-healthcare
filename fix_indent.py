#!/usr/bin/env python3
import re

# Read the file
with open('src/app/ultimate_enhanced2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the Risk Analysis section and fix indentation
result = []
in_risk_section = False
skip_empty = False

for i, line in enumerate(lines):
    # Check if we're entering the Risk Analysis section
    if 'elif selected == "Risk Analysis":' in line:
        in_risk_section = True
        result.append(line)
    # Check if we're exiting the Risk Analysis section
    elif in_risk_section and i > 0 and re.match(r'^    elif selected == ', line):
        in_risk_section = False
        result.append(line)
    # If we're in the Risk Analysis section, add indentation
    elif in_risk_section:
        # Add 4 spaces to non-empty lines that don't already have enough indentation
        if line.strip():  # Non-empty line
            if not line.startswith('        '):  # Less than 8 spaces
                result.append('    ' + line)
            else:
                result.append(line)
        else:  # Empty line
            result.append(line)
    else:
        result.append(line)

# Write back
with open('src/app/ultimate_enhanced2.py', 'w', encoding='utf-8') as f:
    f.writelines(result)

print('Fixed indentation for Risk Analysis section')
