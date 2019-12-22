#!/bin/env python3
import os
import pickle

DATA_DIR = "data/"

print(os.listdir(DATA_DIR))

def load_ds(fname='atis.train.pkl'):
    with open(fname, 'rb') as stream:
        ds, dicts = pickle.load(stream)
    print('Done  loading: ', fname)
    print('      samples: {:4d}'.format(len(ds['query'])))
    print('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
    print('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
    print(' intent count: {:4d}'.format(len(dicts['intent_ids'])))
    return ds, dicts

def write_parallel_data(dset_type, query, slots, intent, i2t, i2s, i2in):
    with open(os.path.join(DATA_DIR, "atis.sentences.%s.csv" % (dset_type)), 'w+') as out:
        for query_idx in query:
            out.write("%s\n" % (','.join(map(i2t.get, query_idx))))
    with open(os.path.join(DATA_DIR, "atis.slots.%s.csv" % (dset_type)), 'w+') as out:
        for slot_ids in slots:
            out.write("%s\n" % (','.join(map(i2s.get, slot_ids))))
    with open(os.path.join(DATA_DIR, "atis.intent.%s.csv" % (dset_type)), 'w+') as out:
        for intent_ids in intent:
            out.write("%s\n" % (','.join(map(i2in.get, intent_ids))))

def create_parallel_dset(ds, dicts, dset_type):
    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids', 'intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]: k for k in d.keys()}, [t2i, s2i, in2i])
    query, slots, intent = map(train_ds.get,
                               ['query', 'slot_labels', 'intent_labels'])
    print ("Writing %s parallel data" % (dset_type))
    write_parallel_data(dset_type, query, slots, intent, i2t, i2s, i2in)

train_ds, dicts = load_ds(os.path.join(DATA_DIR, 'atis.train.pkl'))
create_parallel_dset(train_ds, dicts, 'train')
