import os
import time
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options import Options


if __name__ == '__main__':

    opt = Options().parse()
    opt.isTrain = True

    val_opt = Options().parse(print_options=False)
    val_opt.isTrain = False

    # Fetches the data to train on
    data_loader = create_dataloader(opt, "train_list")

    print('Number of training batches is: %d' % len(data_loader))

    # Store values for training and validation of the model
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.filename, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.filename, "val"))

    # Initializes model instance for training
    model = Trainer(opt)
    
    nb_epoch_patience = 5 # Change this number if you want to wait longer before seeing changes for early stopping
    delta = -0.001 # Change this if you want to consider larger/smaller changes
    early_stopping = EarlyStopping(patience=nb_epoch_patience, delta=delta, verbose=True)

    # Max number of epoch
    nb_epoch = 10000

    # Frequency to save model (every x epochs)
    save_frequency = 20

    # Frequency at which we save the loss for tensorboard
    loss_freq = 100

    for epoch in range(nb_epoch):

        print(f"epoch number: {epoch}")

        epoch_start_time = time.time()
        iter_data_time = time.time()
        
        # The number of images seen in the epoch
        epoch_iter = 0

        for i, data in enumerate(data_loader):

            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()
            
            # Tensorboard display
            if model.total_steps % loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        if epoch % save_frequency == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, model.total_steps))
            model.save_networks('latest')

        # Validation
        model.eval()

        # returns: acc, ap, r_acc, f_acc, f1score, auc_score, prec, recall, y_true, y_pred
        acc, ap, _, _, f1score, roc_score, precision, _, _, _ = validate(model.model, val_opt, "val_list")

        val_writer.add_scalar('f1 Score', f1score, model.total_steps)
        val_writer.add_scalar('ROC Score', roc_score, model.total_steps)
        val_writer.add_scalar('Accuracy', acc, model.total_steps)
        val_writer.add_scalar('Average precision', ap, model.total_steps)
        val_writer.add_scalar('precision', precision, model.total_steps)

        print("Validation at epoch {} | accuracy: {}; average precision: {}".format(epoch, acc, ap))

        # Early stopping based on the average precision achieved
        early_stopping(ap, model)

        if early_stopping.early_stop:

            continue_training = model.adjust_learning_rate()

            # continue_training is True when learning rate doesn't fall under a threshold value

            if continue_training:
                delta_ = -0.002 # Change this if you want to consider larger/smaller changes
                print("Learning rate dropped by 10, training continues...")
                early_stopping = EarlyStopping(patience=nb_epoch_patience, delta=delta_, verbose=True)
            else:
                print("Early stopping, saving the latest model...")
                model.save_networks('latest')
                print("Early stopping.")
                break
        
        # Restarts another epoch
        model.train()
