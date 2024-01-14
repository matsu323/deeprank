fnatfile="Fnat-sorted.dat"
import os
import glob
dir="./data_in/"
moldir=["1ACB"]
mkdir="./ext"
n_divide=100
import shutil
import re
def ext_pdb(name):
    
    # match =re.findall("*.pdb",name)
    # if match :
    #     return match[0]
    # else :return "0"
    sp=name.split(".pdb")
    return sp[0]+".pdb"
for mol in moldir:
    tmp=dir+mol+'/'
    # fnat=glob.glob(tmp+fnatfile)
    # print(fnat)
    with open(tmp+fnatfile)as f:
        file_readline = f.readlines()
    # print(file_readline)
    # for line in file_readline:
        # print (line)


    pdb=file_readline[1::n_divide]
    onlydata=[]
    destination_path=os.path.join(mkdir,mol)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    for i in pdb:
        # onlydata.append(ext_pdb(i))
        source_path=tmp+ext_pdb(i)
        
        try:
            shutil.copy(source_path, destination_path)
        except:
            print("copy error")

    newfn=os.path.join(destination_path,fnatfile)
    print(newfn)
    try:
        with open(newfn, 'w') as file:
            for i in pdb:
                file.write(i)
    except:
        print("dat error")
    # print(onlydata)
    