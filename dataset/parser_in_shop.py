import os.path as osp
import os
from zipfile import ZipFile

root = '../../../datasets/In_shop'
partition = 'Eval/list_eval_partition.txt'

if not osp.isdir(osp.join(root, 'images')):
    os.makedirs(osp.join(root, 'images'))

with open(osp.join(root, partition), 'r') as f:
    partition = f.readlines()

if not osp.isdir(osp.join(root, "Img", "img")):
    print("Extracting zip file")
    with ZipFile(osp.join(root, "Img", "img.zip")) as z:
        z.extractall(os.path.dirname(osp.join(root, "Img", "img.zip")))


train = list()
query = list()
gallery = list()
train_labs = list()
query_labs = list()
gallery_labs = list()

for line in partition:
    line = line.strip()
    if line[:3] != 'img':
        continue
    image_path = line.split(' ')[0]
    class_dir = "{:06d}".format(int(line.split(' ')[-2].split('_')[-1]))
    type = line.split(' ')[-1]

    if not osp.isdir(osp.join(root, 'images', class_dir)):
        os.makedirs(osp.join(root, 'images', class_dir))

    img_new = osp.basename(image_path)
    os.rename(osp.join(root, 'Img', image_path),
              osp.join(root, 'images', class_dir, img_new))

    new_root = os.path.join(*(root.split(os.path.sep)[1:]))
    if type == 'train':
        train.append(osp.join(new_root, 'images', class_dir, img_new) + "\n")
        train_labs.append(str(int(class_dir)) + "\n")
    elif type == 'query':
        query.append(osp.join(new_root, 'images', class_dir, img_new) + "\n")
        query_labs.append(str(int(class_dir)) + "\n")
    elif type == 'gallery':
        gallery.append(osp.join(new_root, 'images', class_dir, img_new) + "\n")
        gallery_labs.append(str(int(class_dir)) + "\n")

with open(osp.join(root, 'train.txt'), 'w') as f:
    f.writelines(train)

with open(osp.join(root, 'train_labs.txt'), 'w') as f:
    f.writelines(train_labs)

with open(osp.join(root, 'query.txt'), 'w') as f:
    f.writelines(query)

with open(osp.join(root, 'query_labs.txt'), 'w') as f:
    f.writelines(query_labs)

with open(osp.join(root, 'gallery.txt'), 'w') as f:
    f.writelines(gallery)

with open(osp.join(root, 'gallery_labs.txt'), 'w') as f:
    f.writelines(gallery_labs)


