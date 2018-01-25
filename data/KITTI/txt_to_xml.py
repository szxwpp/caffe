# encoding: utf-8
from xml.dom.minidom import Document
import os
import cv2


def generate_xml(name, split_lines, img_size, class_ind):
    doc =Document()

    # annotation
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    # annotation->folder
    folder_ele = doc.createElement('folder')
    folder_text = doc.createTextNode('KITTI')
    folder_ele.appendChild(folder_text)
    annotation.appendChild(folder_ele)

    # annotation->filename
    img_name = name + '.png'
    filename_ele = doc.createElement('filename')
    filename_text = doc.createTextNode(img_name)
    filename_ele.appendChild(filename_text)
    annotation.appendChild(filename_ele)

    # annotation->source
    source_ele = doc.createElement('source')
    annotation.appendChild(source_ele)
    # annotation->source->database
    database_ele = doc.createElement('database')
    database_text = doc.createTextNode('The KITTI Database')
    database_ele.appendChild(database_text)
    source_ele.appendChild(database_ele)
    # annotation->source->annotation
    annotation_ele = doc.createElement('annotation')
    annotation_text = doc.createTextNode('KITTI')
    annotation_ele.appendChild(annotation_text)
    source_ele.appendChild(annotation_ele)

    # annotation->size
    size_ele = doc.createElement('size')
    annotation.appendChild(size_ele)
    # annotation->size->width
    width_ele = doc.createElement('width')
    width_text = doc.createTextNode(str(img_size[1]))
    width_ele.appendChild(width_text)
    size_ele.appendChild(width_ele)
    # annotation->size->height
    height_ele = doc.createElement('height')
    height_text = doc.createTextNode(str(img_size[0]))
    height_ele.appendChild(height_text)
    size_ele.appendChild(height_ele)
    # annotation->size->depth
    depth_ele = doc.createElement('depth')
    depth_text = doc.createTextNode(str(img_size[2]))
    depth_ele.appendChild(depth_text)
    size_ele.appendChild(depth_ele)

    for line in split_lines:
        labeldata = line.strip().split()
        if labeldata[0] in class_ind:
            # annotation->object
            object_ele = doc.createElement('object')
            annotation.appendChild(object_ele)

            # annotation->object->name
            name_ele = doc.createElement('name')
            name_text = doc.createTextNode(labeldata[0])
            name_ele.appendChild(name_text)
            object_ele.appendChild(name_ele)

            # annotation->object->bndbox
            bndbox_ele = doc.createElement('bndbox')
            object_ele.appendChild(bndbox_ele)
            # annotation->object->bndbox->xmin
            xmin_ele = doc.createElement('xmin')
            xmin_text = doc.createTextNode(str(int(float(labeldata[4]))))
            xmin_ele.appendChild(xmin_text)
            bndbox_ele.appendChild(xmin_ele)
            # annotation->object->bndbox->ymin
            ymin_ele = doc.createElement('ymin')
            ymin_text = doc.createTextNode(str(int(float(labeldata[5]))))
            ymin_ele.appendChild(ymin_text)
            bndbox_ele.appendChild(ymin_ele)
            # annotation->object->bndbox->xmax
            xmax_ele = doc.createElement('xmax')
            xmax_text = doc.createTextNode(str(int(float(labeldata[6]))))
            xmax_ele.appendChild(xmax_text)
            bndbox_ele.appendChild(xmax_ele)
            # annotation->object->bndbox->ymax
            ymax_ele = doc.createElement('ymax')
            ymax_text = doc.createTextNode(str(int(float(labeldata[7]))))
            ymax_ele.appendChild(ymax_text)
            bndbox_ele.appendChild(ymax_ele)

    with open('Annotations/'+name+'.xml', 'w+') as f:
         f.write(doc.toprettyxml(indent=''))

if __name__ == '__main__':
    class_ind = ('Pedestrian', 'Car', 'Cyclist')
    cur_dir = os.getcwd()
    labels_dir = os.path.join(cur_dir, 'Labels')
    for parent, dimames, filenames in os.walk(labels_dir):
        for file_name in filenames:
            full_path = os.path.join(parent, file_name)
            with open(full_path) as f:
                split_lines = f.readlines()

            img_name = file_name[:-4] + '.png'
            img_path = os.path.join(labels_dir.replace('Labels', 'JPEGImages'), img_name)
            img_size = cv2.imread(img_path).shape

            generate_xml(file_name[:-4], split_lines, img_size, class_ind)

    print('all txts has converted into xmls')

