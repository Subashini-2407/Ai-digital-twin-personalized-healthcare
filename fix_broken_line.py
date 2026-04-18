with open('src/app/ultimate_enhanced3.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Line 2086 (index 2085) has "1 yea\r\nr" - fix it
lines[2085] = '            st.selectbox("Data Retention", ["30 days", "90 days", "1 year"])\n'

with open('src/app/ultimate_enhanced3.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed!')
