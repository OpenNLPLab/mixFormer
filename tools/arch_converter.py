'''
input (argv[1]): architecture stored with dict or yaml format
output (argv[2]): another formatted architecture (different with input)
'''
import sys
import yaml
from glob import glob

files = glob(sys.argv[2])
print(files)

for file in files:
    with open(file, 'r') as fi:
        with open(sys.argv[1], 'w') as fo:
            content = fi.read()
            # input is a dict
            if content.count('\n') < 2:
                arch = eval(content)
                assert isinstance(arch, dict)
                assert 'encoder' in arch.keys()
                assert 'decoder' in arch.keys()
                for module in [arch['encoder'], arch['decoder']]:
                    for k in module:
                        new_k = k.replace('_','-')
                        if 'ffn' in k or 'att' in k:
                            new_k+='-all-subtransformer'
                        else:
                            new_k+='-subtransformer'
                        fo.write('{}:{}\n'.format(new_k, module[k]))
                        print('{}: {}\n'.format(new_k, module[k]))
            else:
                arch_flatten = yaml.safe_load(content)
                arch = dict()
                arch['encoder'] = dict()
                arch['decoder'] = dict()
                for k in arch_flatten.keys():
                    new_k = k.replace('-', '_')
                    new_k = new_k.replace('_all_subtransformer', '')
                    new_k = new_k.replace('_subtransformer', '')
                    if 'encoder' in k:
                        arch['encoder'][new_k] = arch_flatten[k]
                    elif 'decoder' in k:
                        arch['decoder'][new_k] = arch_flatten[k]
                fo.write(str(arch))
                print(arch, ',')