import os
from eval import evaluation
from options.test_options import TestOptions


if __name__ == "__main__":

    opt = TestOptions().parse(print_options=False)
    # eval_resnet50-DDPM-PNDM_bs256
    # /home/abdallah/code/checkpoints/resnet50/
    # resnet50_bs256_DDPM-PNDM/model_epoch_best.pth
    # python testing.py --name eval_resnet50 --batch_size 256 --model_path /home/abdallah/code/checkpoints/resnet50
    
    for dir in os.listdir(opt.model_path):

        """ resnet50_bs256_model1-model2-model3 """
        
        model_path = os.path.join(opt.model_path, dir, "model_epoch_best.pth")
        parts = dir.split('_')  # Split by underscore
        models = parts[-1]  # Get the last part
        _type = str(parts[1:])
        net = parts[0]
    
        name = opt.name + '_' + net + '_' + _type + '_bs' + str(opt.batch_size)

        evaluation(model_path, name, opt)
