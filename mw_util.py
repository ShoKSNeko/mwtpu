""" マルチワーカTPUで使うこまごまとしたコード """
from sys import stderr
from urllib.request import Request, urlopen
from urllib.error import HTTPError

def get_metadata(path):
    requrl = f"http://metadata.google.internal/computeMetadata/v1/{path}"
    headers = {"Metadata-Flavor": "Google"}
    while True:
        try:
            req = Request(requrl, headers=headers)
            with urlopen(req) as s:
                return s.read().decode('utf-8')
        except HTTPError as e:
            if e.code == 503:
                print("metadata server is temporalily unavaliable, retrying...", file=stderr, flush=True)
                sleep(1.)
            else: raise

def my_worker_id():
    return int(get_metadata("instance/attributes/agent-worker-number"))

def num_workers():
    return len(get_metadata("instance/attributes/worker-network-endpoints").split(','))
