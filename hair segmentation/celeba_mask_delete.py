import os
import shutil

PARENT_DIR_OLD = 'CelebAMask-HQ'
PARENT_DIR_NEW = 'dataset'
IMG_DIR_OLD = os.path.join(PARENT_DIR_OLD, 'CelebA-HQ-img')
IMG_DIR_NEW = os.path.join(PARENT_DIR_NEW, 'images')
MASK_DIR_OLD = os.path.join(PARENT_DIR_OLD, 'CelebAMask-HQ-mask-anno')
MASK_DIR_NEW = os.path.join(PARENT_DIR_NEW, 'masks')


def make_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
            print("%s directory has been created." % dir)
        else:
            print("%s directory already exists." % dir)

def move_imgs(img_dir_old, img_dir_new):
    imgs = os.listdir(img_dir_old)
    imgs = [im for im in imgs if im.endswith('.jpg')]
    for im in imgs:
        shutil.copyfile(os.path.join(img_dir_old, im), 
                        os.path.join(img_dir_new, im))
    print("images) %s -> %s finished." % (img_dir_old, img_dir_new))
    return len(imgs)

def move_masks(mask_dir_old, mask_dir_new):
    dirs = os.listdir(mask_dir_old)
    dirs = [os.path.join(mask_dir_old, dir) for dir in dirs 
        if os.path.isdir(os.path.join(mask_dir_old, dir))]
    cnt = 0
    for dir in dirs:
        imgs = os.listdir(dir)
        masks = [im for im in imgs if im.endswith('.png') and 'hair' in im]
        cnt += len(masks)
        for m in masks:
            file_name = str(int(m.split('_')[0])) + '.png'
            shutil.copyfile(os.path.join(dir, m), os.path.join(mask_dir_new, file_name))
    print("masks) %s -> %s finished." % (mask_dir_old, mask_dir_new))
    return cnt

def del_images(img_dir, mask_dir):
    imgs = set(os.listdir(img_dir))
    masks = set(os.listdir(mask_dir))
    cnt = 0
    if len(imgs) > len(masks):
        for im in imgs:
            if im.replace('.jpg', '.png') not in masks:
                os.remove(os.path.join(img_dir, im))
                cnt += 1
    return cnt

def check(img_dir, mask_dir):
    imgs = [im.replace('.jpg', '') for im in os.listdir(img_dir)]
    masks = [m.replace('.png', '') for m in os.listdir(mask_dir)]
    return set(imgs) == set(masks)


if __name__ == "__main__":
    make_dirs([IMG_DIR_NEW, MASK_DIR_NEW])
    images_cnt = move_imgs(IMG_DIR_OLD, IMG_DIR_NEW)
    masks_cnt = move_masks(MASK_DIR_OLD, MASK_DIR_NEW)
    print("len(images): %d, len(masks): %d" % (images_cnt, masks_cnt))
    del_cnt = del_images(IMG_DIR_NEW, MASK_DIR_NEW)
    print("len(deleted images): %d" % del_cnt)
    if check(IMG_DIR_NEW, MASK_DIR_NEW):
        print("Done!")
    else:
        raise Exception("Incompatible images, masks. Aborting...")