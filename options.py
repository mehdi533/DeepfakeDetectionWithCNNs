import argparse
import os
import util
import torch
#import models
#import data


class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
         
        # What model is used to train/test
        parser.add_argument('--arch', type=str, default='res50', help='architecture for binary classification')

        # Architecture options: add a layer before the last one (arXiv:2302.10174v1)
        parser.add_argument('--intermediate', action=argparse.BooleanOptionalAction, type=bool, default=False, help='filename to save')
        parser.add_argument('--intermediate_dim', type=int, default=64, help='Size of the intermediate dimension')

        # Models that should be trained on (e.g: FFpp0,FFpp1 or real,ProGAN...)
        parser.add_argument('--models', default='real', help='models to take into account')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

        # Dataset (dataroot is path to folder containing models, metadata path to folder containing train/val/test lists with subfolders real/fake)
        parser.add_argument('--dataroot', default='./dataset/', help='path to images (should contain folder with all images associated to each image generator)')
        parser.add_argument('--metadata', type=str, default='./dataset/Metadata/', help='directory with list of real/fake images')
        
        # Save options (directory)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpointsMODELS', help='models are saved here')

        # Number of threads depending if using izar (1) or helvetios (4 for faster testing)
        parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')

        # Adjust depending on number of generators training on to have same nb of real/fake images
        parser.add_argument('--multiply_real', type=int, default=1, help='how many times does the real data have to be multiplied by') # Eventually later
        
        # Options for data augmentation
        parser.add_argument('--compression', type=int, default=0)
        parser.add_argument('--blurring', type=int, default=0)

        # Name of the experiment, results save in: checkpoints_dir, filename will be <arch>_<name>_<models>
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

        # parser.add_argument('--generators', default='', help='image generators to train on')
        # parser.add_argument('--class_bal', action='store_true')
        # parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        # parser.add_argument('--filename', default='', help='filename to save')

        # Options for testing

        # Path to model for single testing
        parser.add_argument('--model_path', type=str)

        # Path to folder containing all the models for multi testing
        parser.add_argument('--models_folder_path', type=str, default="./checkpoints")

        self.isTrain = False
        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        # -----------------------------------------------------------------    
        import datetime
        message += 'Date of start: {}\n'.format(datetime.datetime.now())
        # -----------------------------------------------------------------
        message += '----------------- End -------------------'
        print(message)

        # save to the disk

        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mod = "-".join(map(str, opt.models.split(',')[1:]))
        expr_dir = os.path.join(opt.checkpoints_dir, opt.arch+'_'+opt.name+'_'+ mod)

        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        # if opt.suffix:
        #     suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #     opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        # str_ids = opt.gpu_ids.split(',')
        # opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])


        # additional
        # opt.classes = opt.classes.split(',')
        # opt.rz_interp = opt.rz_interp.split(',')
        # opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        # opt.jpg_method = opt.jpg_method.split(',')
        # opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        # if len(opt.jpg_qual) == 2:
        #     opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        # elif len(opt.jpg_qual) > 2:
        #     raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")
        # ------------------------------------------------------------------------
        opt.models = opt.models.split(',')
        mod = "-".join(map(str, opt.models[1:]))
        opt.filename = opt.arch+'_'+opt.name+'_'+ mod
        # ------------------------------------------------------------------------
        self.opt = opt
        return self.opt
