#!/usr/bin/env python3
import re

# Read the file
with open('src/app/ultimate_enhanced2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the Risk Analysis section and fix indentation properly
result = []
in_risk_section = False

for i, line in enumerate(lines):
    # Check if we're entering the Risk Analysis section
    if 'elif selected == "Risk Analysis":' in line and not in_risk_section:
        in_risk_section = True
        result.append(line)
    # Check if we're exiting the Risk Analysis section (entering next elif)
    elif in_risk_section and re.match(r'^    elif selected == ', line) and 'Risk Analysis' not in line:
        in_risk_section = False
        result.append(line)
    # If we're in the Risk Analysis section
    elif in_risk_section and i > 0:
        # Calculate the current indentation level
        current_indent = len(line) - len(line.lstrip()) if line.strip() else 0
        # The first level under elif should be 8 spaces
        # But we're seeing 4 spaces, so we need to add 4
        if line.strip():  # Non-empty line
            if current_indent == 4:
                # This line has 4 spaces but should be indented one more level for at least 8
                result.append('    ' + line)
            else:
                result.append(line)
        else:
            result.append(line)
    else:
        result.append(line)

# Write back
with open('src/app/ultimate_enhanced2.py', 'w', encoding='utf-8') as f:
    f.writelines(result)

print('Fixed indentation for Risk Analysis section (second pass)')
