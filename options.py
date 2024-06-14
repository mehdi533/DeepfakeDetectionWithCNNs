import argparse
import os
import util
from datetime import date


class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Models trained will be saved in this directory')
        parser.add_argument('--name', type=str, default='experiment_name', help='Name of the experiment, results saved in: checkpoints_dir, filename will be <arch>_<name>_<models>')

        # --- What model to train ---
        parser.add_argument('--arch', type=str, default='res50', help='choose the architecture to train for binary classification')

        # --- Architecture options ---
        parser.add_argument('--intermediate', action=argparse.BooleanOptionalAction, type=bool, default=False, help='adds a layer in the classifier (fully connected)')
        parser.add_argument('--intermediate_dim', type=int, default=64, help='Size of the intermediate dimension')
        parser.add_argument("--freeze", action=argparse.BooleanOptionalAction, type=bool, default=False, help='option to freeze the backbone of the model')
        parser.add_argument("--pre_trained", action=argparse.BooleanOptionalAction, type=bool, default=True, help='use the model with pre trained weights')
        parser.add_argument("--cropping", action=argparse.BooleanOptionalAction, type=bool, default=False, help='crop images in random patches')

        # --- Models that should be trained on (e.g: FFpp0,FFpp1 or real,ProGAN...) ---
        parser.add_argument('--models', default='real', help='Format: REALFolder,MOD1,MOD2 models/generators on which the model will be trained')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

        parser.add_argument('--dataroot', default='./dataset/', help='path to images (should contain folder with all images associated to each image generator)')
        parser.add_argument('--metadata', type=str, default='./dataset/Metadata/', help='directory with list of real/fake images')
        parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')

        parser.add_argument('--multiply_real', type=int, default=1, help='how many times does the real data have to be multiplied by') # Eventually later
        
        # --- Options for data augmentation ---
        parser.add_argument('--compr_prob', type=float, default=0.0, help="the percentage of images to be pre processed with compression")
        parser.add_argument('--blur_prob', type=float, default=0.0, help="the percentage of images to be pre processed with compression")

        # --- Options for testing ---
        parser.add_argument('--path', type=str)
        parser.add_argument('--meta_model', type=str, default='average', help='choose between "average" (default), "kNN", "LR": use a meta model for the model ensemble')

        self.isTrain = False
        self.initialized = True

        return parser

    def parse(self, print_options=True):

        opt = self.gather_options()
        
        opt.isTrain = self.isTrain 
        
        # Prints and saves a textfile with the options used in this training
        if print_options:
            self.print_options(opt)

        # The different
        opt.models = opt.models.split(',')
        
        # Creates a list not containing real images for the name of the folder
        tmp_models = []
        for m in opt.models:
            if m == "CelebAHQ" or m == "FFpp0":
                continue
            tmp_models.append(m)
        mod = "-".join(map(str, tmp_models))

        # Formatting of the filename
        opt.filename = opt.arch+'_'+opt.name+'_'+ mod
        
        self.opt = opt

        return self.opt
    
    def gather_options(self):
        
        # Parsing 
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    # Message printed and saved in opt.txt
    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)  
        message += 'The date is: {}\n'.format(date.today())
        message += '----------------- End -------------------'
        print(message)

        mod = "-".join(map(str, opt.models.split(',')[1:]))
        expr_dir = os.path.join(opt.checkpoints_dir, opt.arch+'_'+opt.name+'_'+ mod)

        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

