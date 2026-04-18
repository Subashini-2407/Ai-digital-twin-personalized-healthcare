with open('src/app/ultimate_enhanced3.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the corrupted section - lines 2084-2088 (indices 2083-2087)
lines[2083] = '        with tab3:\n'
lines[2084] = '            st.checkbox("Share anonymized data", False)\n'
lines[2085] = '            st.selectbox("Data Retention", ["30 days", "90 days", "1 year"])\n'
lines[2086] = '\n'
lines[2087] = '    # Footer\n'

with open('src/app/ultimate_enhanced3.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed!')
