import re, os
tex = open('BiasScope_IEEE_Paper.tex', encoding='utf-8').read()

issues = []
if 'PLACEHOLDER' in tex:   issues.append('PLACEHOLDER text still present')
if 'framebox' in tex.lower(): issues.append('framebox still present')
if '[EMAIL_ADDRESS]' in tex: issues.append('Placeholder email still present')
if 'Second Author' in tex:  issues.append('Second Author placeholder not removed')
if 'Third Author' in tex:   issues.append('Third Author placeholder not removed')
if 'EXPAND THIS MORE' in tex: issues.append('Debug comment still present')

figs = re.findall(r'\\includegraphics\[.*?\]\{(.*?)\}', tex)
all_figs_ok = True
for f in figs:
    path = f.replace('../', '')
    exists = os.path.exists(path)
    status = 'EXISTS' if exists else 'MISSING'
    if not exists:
        all_figs_ok = False
    print('  Figure:', path, '->', status)

print()
if not issues and all_figs_ok:
    print('Paper is CLEAN. Ready for submission.')
else:
    for i in issues:
        print('ISSUE:', i)
    if not all_figs_ok:
        print('ISSUE: One or more figure files missing')
