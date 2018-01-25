# encoding:utf-8
# 设置3个类别：Car, Cyclist, Pedestrian
# 将 ‘Van’, ‘Truck’, ‘Tram’ 合并到 ‘Car’ 类别中去，将 ‘Person_sitting’ 合并到 ‘Pedestrian’ 类别中去（‘Misc’ 和 ‘Dontcare’ 这两类直接忽略）

import glob
import string


def show_category(txt_list):
    category_list = []
    for item in txt_list:
        try:
            with open(item) as tdf:
                for each_line in tdf:
                    labeldata = each_line.strip().split(' ')
                    category_list.append(labeldata[0])
        except IOError as ioerr:
            print('File error: '+ str(ioerr))

    print(set(category_list))  # 集合可以去重


def merge2line(labeldata):
    new_line = ''
    for i in range(len(labeldata)-1):
        new_line = new_line + labeldata[i] + ' '
    new_line = new_line + labeldata[len(labeldata)-1] + '\n'

    return new_line


def merge(txt_list):
    for item in txt_list:
        new_txt = []
        try:
            with open(item, 'r') as r_tdf:
                for each_line in r_tdf:
                    labeldata = each_line.strip().split(' ')
                    if labeldata[0] in ['Truck', 'Van', 'Tram']:  # 合并汽车类
                        labeldata[0] = labeldata[0].replace(labeldata[0], 'Car')
                    if labeldata[0] == 'Person_sitting':  # 合并行人类
                        labeldata[0] = labeldata[0].replace(labeldata[0], 'Pedestrian')
                    if labeldata[0] == 'DontCare':  # 忽略Dontcare类
                        continue
                    if labeldata[0] == 'Misc':  # 忽略Misc类
                        continue
                    new_txt.append(merge2line(labeldata))  # 重新写入新的txt文件

            new_item = item.replace('Labels_ori', 'Labels')
            with open(new_item, 'w+') as w_tdf:  # w+是打开原文件将内容删除，另写新内容进去
                for temp in new_txt:
                    w_tdf.write(temp)

        except IOError as ioerr:
            print('File error: ' + str(ioerr))


txt_list = glob.glob('./Labels_ori/*.txt')
print('before modify categories are:\n')
show_category(txt_list)

merge(txt_list)

print('after modify categories are:\n')
modify_txt_list = glob.glob('./Labels/*.txt')
show_category(modify_txt_list)



