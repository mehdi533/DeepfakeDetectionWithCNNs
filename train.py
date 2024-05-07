import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions


def get_val_opt():
    # Takes same default options as training but modifies them for validation purposes
    val_opt = TrainOptions().parse(print_options=False)
<<<<<<< HEAD

=======
    # ---------------------------------------- ----------------------------------------
    #val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    # ---------------------------------------- ----------------------------------------
>>>>>>> updated_code
    # val_opt.isTrain = False
    # val_opt.no_resize = False
    # val_opt.no_crop = False
    # val_opt.serial_batches = True
<<<<<<< HEAD

    # Difference between pil and cv2?
    # val_opt.jpg_method = ['pil']

    # Wtf is this, why would you apply blurring in validation or compression??
=======
    # val_opt.jpg_method = ['pil']
>>>>>>> updated_code
    # if len(val_opt.blur_sig) == 2:
    #     b_sig = val_opt.blur_sig
    #     val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    # if len(val_opt.jpg_qual) != 1:
    #     j_qual = val_opt.jpg_qual
    #     val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':

    opt = TrainOptions().parse()

    val_opt = get_val_opt()

    # Check what are the models to train on
    data_loader = create_dataloader(opt, "train_list")

    print('Number of training batches is: %d' % len(data_loader))

    # Store values for training and validation of the model
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.filename, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.filename, "val"))

    # Initializes model instance for training
    model = Trainer(opt)
<<<<<<< HEAD
    # Early stop implementation
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, verbose=True, delta=-0.001)

    for epoch in range(opt.nepoch):

        print(f"Starting epoch number: {epoch}")
=======
    # early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    early_stopping = EarlyStopping(patience=5, delta=-0.001, verbose=True)
    
    nb_epoch = 10000
    # for epoch in range(opt.niter):
    for epoch in range(nb_epoch):

        print(f"epoch number: {epoch}")

>>>>>>> updated_code
        epoch_start_time = time.time()
        iter_data_time = time.time()
        
        # The number of images seen in the epoch
        epoch_iter = 0

        for i, data in enumerate(data_loader):

            model.total_steps += 1
            epoch_iter += opt.batch_size

<<<<<<< HEAD
            # Sends image and label to device (gpu)
=======
>>>>>>> updated_code
            model.set_input(data)

            model.optimize_parameters()
            
            # Tensorboard display
            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

<<<<<<< HEAD
            # Unnecessary
    
=======
>>>>>>> updated_code
            # if model.total_steps % opt.save_latest_freq == 0:
            #     print('saving the latest model %s (epoch %d, model.total_steps %d)' %
            #           (opt.name, epoch, model.total_steps))
            #     model.save_networks('latest')
<<<<<<< HEAD
=======

            # print("Iter time: %d sec" % (time.time()-iter_data_time))
            # iter_data_time = time.time()
>>>>>>> updated_code

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            # model.save_networks(epoch)

        # Validation
        model.eval()
<<<<<<< HEAD
=======
        # # ---------------------------------------- ----------------------------------------
        # # acc, ap = validate(model.model, val_opt, "val_list")[:2]
        # acc, ap , _, _, f1score, auc_score, _, _, _, _ = validate(model.model, val_opt, "val_list")
        # val_writer.add_scalar('f1 score', f1score, model.total_steps)
        # val_writer.add_scalar('AUC score', auc_score, model.total_steps)
        # # ---------------------------------------- ----------------------------------------
        # val_writer.add_scalar('accuracy', acc, model.total_steps)
        # val_writer.add_scalar('ap', ap, model.total_steps)
        # print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        
        # early_stopping(acc, model)

>>>>>>> updated_code

        # returns: acc, ap, r_acc, f_acc, f1score, auc_score, prec, recall, y_true, y_pred
        acc, ap, _, _, f1score, roc_score, precision, _, _, _ = validate(model.model, val_opt, "val_list")

        val_writer.add_scalar('f1 Score', f1score, model.total_steps)
        val_writer.add_scalar('ROC Score', roc_score, model.total_steps)
        val_writer.add_scalar('Accuracy', acc, model.total_steps)
        val_writer.add_scalar('Average precision', ap, model.total_steps)
        val_writer.add_scalar('precision', precision, model.total_steps)

        print("Validation at epoch {} | accuracy: {}; average precision: {}".format(epoch, acc, ap))

<<<<<<< HEAD
        # Early stopping based on the accuracy
        early_stopping(acc, model)
=======
        # Early stopping based on the average precision achieved
        early_stopping(ap, model)

        # if early_stopping.early_stop:
        #     cont_train = model.adjust_learning_rate()
        #     if cont_train:
        #         print("Learning rate dropped by 10, continue training...")
        #         early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
        #     else:
        #         print("Early stopping.")
        #         break
        # model.train()

>>>>>>> updated_code

        if early_stopping.early_stop:
            continue_training = model.adjust_learning_rate()

            # continue_training is True when learning rate doesn't fall under a threshold value
            if continue_training:
                print("Learning rate dropped by 10, training continues...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        
        # Restarts another epoch
        model.train()
