import torch
from model import model, transfer_model
from loss.loss import iou_loss
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils.custom_transfrom import UnNormalize

class Tester:
    def __init__(self, config, dataloader):
        self.batch_size = config.batch_size
        self.config = config
        self.model_path = config.checkpoint_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = dataloader
        self.image_len = len(dataloader)
        self.num_classes = config.num_classes
        self.num_test = config.num_test
        self.sample_dir = config.sample_dir
        self.sample_step = config.sample_step
        self.epoch = config.epoch
        self.transfer_learning = config.transfer_learning
        self.build_model()

    def build_model(self):
        if self.transfer_learning:
            self.net = transfer_model.MobileHairNet().to(self.device)
        else:
            self.net = model.MobileHairNet().to(self.device)
        self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.model_path))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.listdir(self.model_path):
            print("[!] No checkpoint in ", str(self.model_path))
            return

        model_path = os.path.join(self.model_path, f"MobileHairNet_epoch-{self.epoch-1}.pth")
        model = glob(model_path)
        model.sort()
        if not model:
            raise Exception(f"[!] No Checkpoint in {model_path}")

        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print(f"[*] Load Model from {model[-1]}: ")

    def test(self):
        self.net.eval()
        with torch.no_grad():
            unnormal = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            for step, (image, mask) in enumerate(self.data_loader):
                image = image.to(self.device)
                mask = mask.to(self.device)
                result = self.net(image).to(self.device)
                image = unnormal(image.to(self.device))
                iou = iou_loss(result, mask)

                # save sample images
                if step % 50 == 0:
                    print(f"Step: [{step}/{self.image_len}] | IOU: {iou:.4f}")
                if step % self.sample_step == 0:
                    self.save_sample_imgs(image[0], mask[0], torch.argmax(result[0], 0), self.sample_dir, step)
                    print('[*] Saved sample images')

    def save_sample_imgs(self, real_img, real_mask, prediction, save_dir, step):
        data = [real_img, real_mask, prediction]
        names = ["Image", "Mask", "Prediction"]

        fig = plt.figure()
        for i, d in enumerate(data):
            d = d.squeeze()
            im = d.data.cpu().numpy()

            if i > 0:
                im = np.expand_dims(im, axis=0)
                im = np.concatenate((im, im, im), axis=0)

            im = (im.transpose(1, 2, 0) + 1) / 2

            f = fig.add_subplot(1, 3, i + 1)
            f.imshow(im)
            f.set_title(names[i])
            f.set_xticks([])
            f.set_yticks([])

        p = os.path.join(save_dir, "Test_step-%s.png" % (step))
        plt.savefig(p)