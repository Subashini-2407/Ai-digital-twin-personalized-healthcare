from pathlib import Path
path=Path('src/app/ultimate_enhanced3.py')
text=path.read_text(encoding='utf-8')
count = text.count('"""') + text.count("'''")
print('total triple quote tokens:', count)
print('parity:', count % 2)
for i,raw in enumerate(text.splitlines(),1):
    if raw.count('"""') + raw.count("'''") > 0:
        print('triple quotes at', i, raw.strip()[:120])
paren=0
brack=0
brace=0
for i,raw in enumerate(text.splitlines(),1):
    for ch in raw:
        if ch=='(': paren += 1
        elif ch==')': paren -= 1
        elif ch=='[': brack += 1
        elif ch==']': brack -= 1
        elif ch=='{': brace += 1
        elif ch=='}': brace -= 1
    if paren<0 or brack<0 or brace<0:
        print('negative depth at', i, paren, brack, brace)
        break
print('final depth', paren, brack, brace)
