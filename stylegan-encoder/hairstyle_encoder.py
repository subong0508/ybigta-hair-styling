#from PIL import Image
import os

'''
os.chdir('C:\\Users\icarus\PycharmProjects\Hairstyle-Transfer')
imgs = sorted(os.listdir('raw_images'))

print("Found %d images in %s" % (len(imgs), 'raw_images'))
if len(imgs) == 0:
  print("Upload images to the \"raw_images\" folder!")
else:
  print(imgs)

for img_path in imgs:
  img = Image.open('raw_images/' + img_path)

  w, h = img.size
  rescale_ratio = 256 / min(w, h)
  img = img.resize((int(rescale_ratio * w), int(rescale_ratio * h)), Image.LANCZOS)
  img.show()
  '''

os.chdir('/home/yonsei/ryan0507/stylegan-encoder')
print("aligned_images contains %d images ready for encoding!" %len(os.listdir('aligned_images/')))
print("Recommended batch_size for the encode_images process: %d" %min(len(os.listdir('aligned_images/')), 8))
