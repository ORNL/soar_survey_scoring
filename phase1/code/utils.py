#########   General functions   ###########
""" Includes functions for serialization (jsonify, unjsonify, picklify, unpickle) and
finding all locations of a given element in a list"""

import codecs
import datetime
import json
import os
import pickle
import sys
from collections import defaultdict

from elasticsearch import Elasticsearch

TEST_NAMES_BLACKLIST = ['test', 'TEST', '_failed',
                        'REDUX2_Exploit_Move_ETERNALBLUE', 'REDUX_Exploit_Move_ETERNALBLUE',
                        'TaggerLE_CVE-2006-4437_meterpreter_pivot_beacon_1024']

# Parsing specific functions


def not_implemented(d):
    return "not implemented"


def tbd(d):
    return "tbd"


def blacklist_name_substrings(substrings):
    def blacklist(d):
        return any(substring in d['attack_name'] for substring in substrings)

    return blacklist


def get_blacklist_conditions():
    blacklist_conditions = [
        blacklist_name_substrings(TEST_NAMES_BLACKLIST),
    ]

    return blacklist_conditions


# Get unique set of IPs from attacks
def get_unique_ips_from_attacks(start_time, end_time):
    es, res = get_attacks(start_time, end_time)
    blacklist_conditions = get_blacklist_conditions()

    ip_set = set()
    for hit_num in range(0, len(res['hits']['hits'])):
        attack_id = res['hits']['hits'][hit_num]['_id']
        d = res['hits']['hits'][hit_num]['_source']
        # By default filter out tests
        if any(blacklist_condition(d) for blacklist_condition in blacklist_conditions):
            continue
        ip_set.update(d['ips'])

    return ip_set


# ElasticSearch specific functions for getting attacks
def get_attacks(start_time, end_time):
    es = Elasticsearch()
    es.indices.refresh(index='attacks')
    search_body = {
        "query":
            {"range":
                 {"start_time":
                      {"gte": start_time,
                       "lte": end_time}
                  }
             }
    }

    res = es.search(index='attacks', body=search_body, size=10000)

    return es, res


# functions for saving/opening objects
def jsonify(obj, out_file):
    """
    Inputs:
    - obj: the object to be jsonified
    - out_file: the file path where obj will be saved
    This function saves obj to the path out_file as a json file.
    """
    json.dump(obj, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def unjsonify(in_file):
    """
    Input:
    -in_file: the file path where the object you want to read in is stored
    Output:
    -obj: the object you want to read in
    """
    obj_text = codecs.open(in_file, 'r', encoding='utf-8').read()
    obj = json.loads(obj_text)
    return obj

def keystoint(x):
    try:
        return {int(k): v for k, v in x.items()}
    except:
        return x
        
def unjsonify_int_keys(in_file):
    obj_text = codecs.open(in_file, 'r', encoding='utf-8').read()
    obj = json.loads(obj_text, object_hook=keystoint)
    return obj

def picklify(obj, filepath):
    """
    Inputs:
    - obj: the object to be pickled
    - filepath: the file path where obj will be saved
    This function pickles obj to the path filepath.
    """
    pickle_file = open(filepath, "wb")
    pickle.dump(obj, pickle_file)
    pickle_file.close()
    # print "picklify done"


def unpickle(filepath):
    """
    Input:
    -filepath: the file path where the pickled object you want to read in is stored
    Output:
    -obj: the object you want to read in
    """
    pickle_file = open(filepath, 'rb')
    obj = pickle.load(pickle_file)
    pickle_file.close()
    return obj


def curtime_str():
    """A string representation of the current time."""
    dt = datetime.datetime.now().time()
    return dt.strftime("%H:%M:%S")


def update_json_dict(key, value, out_file, overwrite=True):
    if not os.path.isfile(out_file):
        d = {}
    else:
        d = unjsonify(out_file)
        if key in d and not overwrite:
            print("fkey {key} already in {out_file}, skipping...")
            return
    d[key] = value
    jsonify(d, out_file)


def update_json_dict_nested(value, out_file, key1, key2=None, overwrite=True):
    if not os.path.isfile(out_file):
        d = {}
    else:
        d = unjsonify(out_file)

    if key1 in d:
        if key2 in d[key1] and not overwrite:
            print("fkey {key1} already in {out_file}, skipping...")
            return
        d[key1][key2] = value
    else:
        d[key1] = {key2: value}

    jsonify(d, out_file)

def es_infinite_scroll(es, index, body):
    resp = es.search(index=index, body=body, scroll='1m')
    scroll_id = resp['_scroll_id']
    while len(resp['hits']['hits']) > 0:
        for hit_num in range(0, len(resp['hits']['hits'])):
            yield resp['hits']['hits'][hit_num]
        resp = es.scroll(scroll_id=scroll_id, scroll='1m')
