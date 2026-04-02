from mw_util import get_metadata, my_worker_id, num_workers

'''
def listdir(path):
    print(f"{path}:")
    for ent in get_metadata(path).split():
        print(f" {ent}")
'''

print(f"my_worker_id = {my_worker_id()} out of {num_workers()} workers")
'''
print()
listdir("instance/guest-attributes/")
print()
listdir("instance/guest-attributes/deviceInfo/")
print()
listdir("instance/guest-attributes/guest-agent/")
print()
listdir("instance/guest-attributes/hostkeys/")
print()
listdir("instance/guest-attributes/ia/")
print()
listdir("instance/guest-attributes/lifecycle/")
'''
#print(get_metadata("instance/attributes/worker-network-endpoints"))
