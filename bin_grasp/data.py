"""Load compact frozen features; keep answers and metadata outside model inputs."""
import json
import math
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, index_path):
        self.path = Path(index_path)
        self.index = json.loads(self.path.read_text())
        self.rows = self.index['rows']
        self.texts = np.load(self.path.parent / self.index['text'], mmap_mode='r')
        if self.texts.shape != (len(self.rows), 768):
            raise ValueError('Text feature shape does not match index')
        self.scene_cache = OrderedDict()
        self.classes = {}
        for r in self.rows:
            self.classes.setdefault(r['scene'], {})[r['target_id']] = r['target_class']

    def __len__(self):
        return len(self.rows)

    def scene_record(self, scene):
        if scene in self.scene_cache:
            self.scene_cache.move_to_end(scene)
            return self.scene_cache[scene]
        data = torch.load(self.path.parent / self.index['scenes'][scene], weights_only=True, map_location='cpu')
        self.scene_cache[scene] = data
        if len(self.scene_cache) > 256:
            self.scene_cache.popitem(last=False)
        return data

    def __getitem__(self, i):
        row = self.rows[i]
        data = self.scene_record(row['scene'])
        if row['target_id'] not in data['ids']:
            raise ValueError(f'Target missing: {row["key"]}')
        known = self.classes[row['scene']]
        duplicate = sum(known.get(k) == row['target_class'] for k in data['ids']) > 1
        inputs = dict(visual=data['visual'].float(), scene=data['scene'].float(),
                      text=torch.from_numpy(np.array(self.texts[i], dtype=np.float32)), geometry=data['geometry'].float())
        return inputs, data['ids'].index(row['target_id']), dict(
            key=row['key'], sentence=row['sentence'], duplicate=duplicate,
            class_coverage_complete=all(k in known for k in data['ids']),
            candidate_ids=data['ids'])


def collate(samples):
    count = len(samples)
    maximum = max(len(s[0]['visual']) for s in samples)
    dim = samples[0][0]['visual'].shape[-1]
    batch = dict(visual=torch.zeros(count, maximum, dim), geometry=torch.zeros(count, maximum, 7),
                 text=torch.stack([s[0]['text'] for s in samples]),
                 scene=torch.stack([s[0]['scene'] for s in samples]), valid=torch.zeros(count, maximum, dtype=torch.bool))
    for i, (inputs, _, _) in enumerate(samples):
        n = len(inputs['visual'])
        batch['visual'][i, :n] = inputs['visual']
        batch['geometry'][i, :n] = inputs['geometry']
        batch['valid'][i, :n] = True
    return batch, torch.tensor([s[1] for s in samples]), [s[2] for s in samples]


def verify_pair(train, val):
    for key in ('cache_id', 'mode', 'fingerprint'):
        if train.index[key] != val.index[key]:
            raise ValueError(f'Train/validation mismatch: {key}')
    if train.index['split'] != 'train' or val.index['split'] != 'val':
        raise ValueError('Training requires train and val feature indexes; never use test to select a model')
    if {r['key'] for r in train.rows} & {r['key'] for r in val.rows}:
        raise ValueError('Training and validation samples overlap')
    if train.index['mode'] == 'sequence':
        if {r['group'] for r in train.rows} & {r['group'] for r in val.rows}:
            raise ValueError('Sequence leakage detected')


class ResidentFeatures:
    """Read each scene once, then gather whole batches from RAM or GPU tensors."""
    def __init__(self, dataset, storage='auto', device='cuda'):
        self.dataset = dataset
        scenes = sorted({r['scene'] for r in dataset.rows})
        records = [dataset.scene_record(s) for s in scenes]
        scene_index = {s:i for i,s in enumerate(scenes)}
        maximum = max(len(r['ids']) for r in records)
        width = records[0]['visual'].shape[-1]
        inputs = dict(visual=torch.zeros(len(scenes),maximum,width),
                      geometry=torch.zeros(len(scenes),maximum,7),
                      scene=torch.stack([r['scene'].float() for r in records]),
                      valid=torch.zeros(len(scenes),maximum,dtype=torch.bool))
        id_maps = []
        catalogs = []
        for i,(scene,record) in enumerate(zip(scenes,records)):
            n = len(record['ids'])
            inputs['visual'][i,:n] = record['visual']
            inputs['geometry'][i,:n] = record['geometry']
            inputs['valid'][i,:n] = True
            id_maps.append({k:j for j,k in enumerate(record['ids'])})
            known = dataset.classes[scene]
            counts = {}
            for k in record['ids']:
                if k in known:
                    counts[known[k]] = counts.get(known[k],0)+1
            catalogs.append((counts,all(k in known for k in record['ids'])))
        row_scenes = torch.tensor([scene_index[r['scene']] for r in dataset.rows])
        targets = torch.tensor([id_maps[scene_index[r['scene']]][r['target_id']] for r in dataset.rows])
        texts = torch.from_numpy(np.array(dataset.texts,dtype=np.float32))
        self.metadata = []
        for row in dataset.rows:
            i = scene_index[row['scene']]
            counts, complete = catalogs[i]
            self.metadata.append(dict(key=row['key'],sentence=row['sentence'],
                duplicate=counts.get(row['target_class'],0)>1,
                class_coverage_complete=complete,candidate_ids=records[i]['ids']))
        tensors = list(inputs.values())+[row_scenes,targets,texts]
        self.bytes = sum(t.numel()*t.element_size() for t in tensors)
        location = 'cpu'
        if storage == 'cuda' and device != 'cuda':
            raise ValueError('--feature-storage cuda requires --device cuda')
        if device == 'cuda' and storage != 'cpu':
            free,_ = torch.cuda.mem_get_info()
            if self.bytes + 3*1024**3 < free:
                location = 'cuda'
            elif storage == 'cuda':
                raise RuntimeError('Insufficient GPU headroom for resident features; use --feature-storage cpu')
        self.location = location
        self.inputs = {k:v.to(location) for k,v in inputs.items()}
        self.texts = texts.to(location)
        self.row_scenes = row_scenes.to(location)
        self.targets = targets.to(location)
        dataset.scene_cache.clear()
        print(f'Preloaded {len(dataset)} expressions / {len(scenes)} scenes: '
              f'{self.bytes/1024**3:.2f} GiB on {location.upper()}',flush=True)

    def loader(self,batch_size,shuffle=False,metadata=True):
        return ResidentLoader(self,batch_size,shuffle,metadata)


class ResidentLoader:
    def __init__(self,features,batch_size,shuffle,metadata):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.features,self.batch_size,self.shuffle,self.metadata = features,batch_size,shuffle,metadata

    def __len__(self):
        return math.ceil(len(self.features.dataset)/self.batch_size)

    def __iter__(self):
        f = self.features
        count = len(f.dataset)
        cpu_order = torch.randperm(count) if self.shuffle else torch.arange(count)
        order = cpu_order.to(f.location)
        for start in range(0,count,self.batch_size):
            rows = order[start:start+self.batch_size]
            scenes = f.row_scenes[rows]
            inputs = {k:v[scenes] for k,v in f.inputs.items()}
            inputs['text'] = f.texts[rows]
            meta = [f.metadata[i] for i in cpu_order[start:start+self.batch_size].tolist()] if self.metadata else []
            yield inputs,f.targets[rows],meta
