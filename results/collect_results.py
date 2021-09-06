import json
import numpy as np
from collections import defaultdict
import pandas as pd


if __name__ == '__main__':
    dic = defaultdict(list)
    for s in ['eedi', 'elemmath_2021', 'ednet_kt3', 'junyi_15']:
        for m in ['DKT2', 'SAKT', 'SAINT', 'SAINT_Plus', 'SAINT_Plus_Features']:
            try:
                d = json.load(open(f'{m}_{s}.json', 'r'))
                for o, sub_d in d.items():
                    dic['dataset'].append(s)
                    dic['model'].append(m)
                    dic['mode'].append(o)
                    for k, v in sub_d.items():
                        dic[k].append(np.mean(v))
                        dic[f'{k}_std'].append(np.std(v))
            except:
                pass

    pd.DataFrame.from_dict(dic).to_csv('res.csv', index=False)
