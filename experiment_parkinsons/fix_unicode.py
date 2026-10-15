"""Fix unicode characters that Windows cp950 console cannot encode."""
import sys

path = r'c:\Users\yuwen\Downloads\1014-hw2\1014-hw2\experiment_parkinsons\experiment_parkinsons.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    ('\u00b2', '^2'),
    ('\u00b1', '+/-'),
    ('\u2192', '->'),
    ('\u2191', '(up)'),
    ('\u2193', '(dn)'),
    ('\u2013', '-'),
    ('\u2014', '--'),
    ('\u2022', '*'),
    ('\u2713', 'v'),
    ('\u2714', 'v'),
    ('\u271a', '+'),
    ('\u00e9', 'e'),
    ('\u2018', "'"),
    ('\u2019', "'"),
    ('\u201c', '"'),
    ('\u201d', '"'),
]
for old, new in replacements:
    content = content.replace(old, new)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
bad = [(i+1, ord(c), repr(c)) for i, c in enumerate(content) if ord(c) > 127]
print(f'Remaining non-ASCII chars: {len(bad)}')
for pos, cp, ch in bad[:30]:
    print(f'  pos {pos}: U+{cp:04X} {ch}')
