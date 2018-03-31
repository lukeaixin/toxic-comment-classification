import re

def normalize(txt):
	txt = txt.strip().lower()
	txt = re.sub(r'[^\w\s]+', ' ', txt)
	txt = re.sub(r'_+', ' ', txt)
	txt = re.sub(r'\d+', ' ', txt)
	txt = ' '.join(txt.split())
	return txt

