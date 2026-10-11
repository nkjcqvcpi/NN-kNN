import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parent / 't3-public-data-20261005'
root.mkdir(exist_ok=False)
revision = 'eee5fdbf62f8613a7812b03419e6b29617b74fd1'
sources = [
    ('squad_train_v11.json',f'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{revision}/dataset/train-v1.1.json','squad',87599),
    ('squad_dev_v11.json',f'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{revision}/dataset/dev-v1.1.json','squad',10570),
    ('hotpot_dev_distractor_v1.json','https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json','hotpot',7405),
    ('hotpot_evaluate_v1.py','https://raw.githubusercontent.com/hotpotqa/hotpot/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py','code',None),
    ('squad_evaluate_v11.py',f'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{revision}/evaluate-v1.1.py','code',None),
]
records = []
for name,url,kind,count in sources:
    print('downloading',name,flush=True)
    path = root/name
    digest = hashlib.sha256()
    with urllib.request.urlopen(url,timeout=60) as response, path.open('xb') as file:
        final_url = response.url
        for chunk in iter(lambda:response.read(1024*1024),b''):
            file.write(chunk); digest.update(chunk)
    if kind != 'code':
        obj = json.loads(path.read_text(encoding='utf-8'))
        actual = len(obj) if kind == 'hotpot' else sum(len(p['qas']) for a in obj['data'] for p in a['paragraphs'])
        assert actual == count, (name,actual,count)
    else:
        actual = None
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
    records.append(dict(name=name,url=url,final_url=final_url,sha256=digest.hexdigest(),bytes=path.stat().st_size,examples=actual))
    (root/'manifest.json').write_text(json.dumps(dict(files=records,complete=len(records)==len(sources),
        dataset_license='CC BY-SA 4.0 according to both official dataset sites',
        scope='SQuAD1.1 train/dev and HotpotQA distractor dev only; no Hotpot train/fullwiki/test download'),indent=2),encoding='utf-8')
    print('validated',name,path.stat().st_size,actual,flush=True)
