import os.path as osp
import os

root = '../../../datasets/Stanford_Online_Products'
train = 'Ebay_train.txt'
test = 'Ebay_test.txt'

if not osp.isdir(osp.join(root, 'images')):
    os.makedirs(osp.join(root, 'images'))

with open(osp.join(root, train), 'r') as f:
    train_labels = f.readlines()

with open(osp.join(root, test), 'r') as f:
    test_labels = f.readlines()

for data in [train_labels, test_labels]:
    for line in data:
        line = line.strip()
        if line.split(' ')[0] == 'image_id':
            continue

        image_path = line.split(' ')[3]
        class_dir = "{:05d}".format(int(line.split(' ')[1]))

        if not osp.isdir(osp.join(root, 'images', class_dir)):
            os.makedirs(osp.join(root, 'images', class_dir))

        img_new = osp.basename(image_path)
        os.rename(osp.join(root, image_path),
                  osp.join(root, 'images', class_dir,img_new))


