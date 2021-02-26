import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--model_hair_path', type=str, default='.\hairsegmentation\checkpoints')
parser.add_argument('--model_hair_weights', type=str, default='MobileHairNet_hair_epoch-198.pth')
parser.add_argument('--model_skin_path', type=str, default='.\skinsegmentation\checkpoints')
parser.add_argument('--model_skin_weights', type=str, default='MobileHairNet_skin_epoch-198.pth')
parser.add_argument('--freeze', type=bool, default=True)
parser.add_argument('--image_path', type=str, default='.\hairsegmentation\dataset\images')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--checkpoints', type=str, default='.\checkpoints')
parser.add_argument('--save_path', type=str, default='.\\samples\\results')
parser.add_argument('--p', type=float, default=0.9)
parser.add_argument('--lambda_gan', type=float, default=100)
parser.add_argument('--lambda_hair', type=float, default=10)
parser.add_argument('--lambda_face', type=float, default=1)
parser.add_argument('--lambda_style', type=float, default=1e5)
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument('--your_pic',type=str,help='The absolute path of your image')
parser.add_argument('--celeb_pic',type=str,help='The absolute path of the celebrity image you wish to synthesize to your picture.')

def get_config():
    args = parser.parse_args()
    your_pic = args.your_pic
    celeb_pic = args.celeb_pic
    return args