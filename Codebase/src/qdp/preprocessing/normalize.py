import re, unicodedata

def normalize_working_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).replace('\r\n','\n').replace('\r','\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return '\n'.join(line.rstrip() for line in text.split('\n')).strip()

def duplicate_norm(text: str) -> str:
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text)).strip().casefold()
