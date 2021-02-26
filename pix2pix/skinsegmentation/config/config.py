import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--image_size', type=int, default=255, help='the height / width of the input image to network')
parser.add_argument('--num_classes', type=int, default=2, help='number of model output channels')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
parser.add_argument('--epoch', type=int, default=0, help='start epochs to train for')
parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--decay_epoch', type=int, default=100, help='learning rate decay start epoch num')
parser.add_argument('--lr', type=float, default=1, help='learning rate')
parser.add_argument('--rho', type=float, default=0.95, help='adadelta rho')
parser.add_argument('--eps', type=float, default=1e-7, help='adadelta eps')
parser.add_argument('--decay', type=float, default=2e-5, help='adadelta decay')
parser.add_argument('--sample_step', type=int, default=100, help='step of saving sample images')
parser.add_argument('--checkpoint_step', type=int, default=100, help='step of saving checkpoints')
parser.add_argument('--data_path', default='./dataset', help='path to dataset')
parser.add_argument('--test_data_path', default='./dataset', help='path to test dataset')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--sample_dir', default='samples', help='folder to output images and model checkpoints')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--mode', type=str, default='train', help='Trainer mode: train or test')
parser.add_argument('--num_test', type=int, default=32, help='The number of test image')
parser.add_argument('--transfer_learning', type=bool, default=True, help='whether use pretrained mobilenet v2')
parser.add_argument('--gradient_loss_weight', type=int, default=0.5, help='The number of test image')

def get_config():
    return parser.parse_args()
