import sys
import os

home_path=os.environ['HOME']
data_path=os.path.join(home_path,'github/data/mtcnn')

stdsize = 12
save_dir = os.path.join(data_path,str(stdsize))
if not os.path.exists(save_dir):
    raise "directory is not exist"

f1 = open(os.path.join(save_dir, 'pos_%s.txt'%str(stdsize)), 'r')
f2 = open(os.path.join(save_dir, 'neg_%s.txt'%str(stdsize)), 'r')
f3 = open(os.path.join(save_dir, 'part_%s.txt'%str(stdsize)), 'r')

pos = f1.readlines()
neg = f2.readlines()
part = f3.readlines()
f = open(os.path.join(save_dir, 'label-train.txt'), 'w')

for i in range(int(len(pos))):
    p = pos[i].find(" ") + 1
    pos[i] = pos[i][:p-1] + ".jpg " + pos[i][p:-1] + "\n"
    f.write(pos[i])

for i in range(int(len(neg))):
    p = neg[i].find(" ") + 1
    neg[i] = neg[i][:p-1] + ".jpg " + neg[i][p:-1] + " -1 -1 -1 -1\n"
    f.write(neg[i])

for i in range(int(len(part))):
    p = part[i].find(" ") + 1
    part[i] = part[i][:p-1] + ".jpg " + part[i][p:-1] + "\n"
    f.write(part[i])

f1.close()
f2.close()
f3.close()
