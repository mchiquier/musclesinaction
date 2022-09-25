'''
Evaluation tools.
'''


# Internal imports.
import args
import data
import logvis
import loss
import model
import pipeline
import utils


def load_networks(checkpoint_path, device, logger, epoch=-1):
    '''
    :param checkpoint_path (str): Path to model checkpoint folder or file.
    :param epoch (int): If >= 0, desired checkpoint epoch to load.
    :return (networks, train_args, dset_args, model_args, epoch).
        networks (list of modules).
        train_args (dict).
        train_dset_args (dict).
        model_args (dict).
        epoch (int).
    '''
    print_fn = logger.info if logger is not None else print
    assert os.path.exists(checkpoint_path)
    if os.path.isdir(checkpoint_path):
        model_fn = f'model_{epoch}.pth' if epoch >= 0 else 'checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_path, model_fn)

    print_fn('Loading weights from: ' + checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Display all arguments to help verify correctness.
    train_args = checkpoint['train_args']
    train_dset_args = checkpoint['dset_args']
    print_fn('Train command args: ' + str(train_args))
    print_fn('Train dataset args: ' + str(train_dset_args))

    # Get network instance parameters.
    model_args = checkpoint['model_args']
    print_fn('My model args: ' + str(model_args))

    # Instantiate network.
    my_model = model.MyModel(logger, **model_args)
    my_model = my_model.to(device)
    my_model.load_state_dict(checkpoint['my_model'])

    networks = [my_model]

    epoch = checkpoint['epoch']
    print_fn('=> Loaded epoch (1-based): ' + str(epoch + 1))

    return (networks, train_args, train_dset_args, model_args, epoch)


def perform_inference(data_retval, networks, device, logger, all_args):
    '''
    Generates test time predictions.
    :param all_args (dict): train, test, train_dset, test_dest, model.
    '''
    print_fn = logger.info if logger is not None else print

    # Prepare model.
    my_model = networks[0]

    # Prepare data.
    # TODO: Inputs can sometimes be specially constructed / controlled for, and may not be a dict
    # stemming from the typical data loader. We can optionally write code to accommodate any custom
    # input formats here.
    rgb_input = data_retval['rgb_input'].to(device)
    (B, C, H, W) = rgb_input.shape

    # Run model.
    rgb_output = my_model(rgb_input)
    
    # Convert data from torch to numpy for futher processing.
    rgb_input = rearrange(rgb_input, 'B C H W -> B H W C').detach().cpu().numpy()
    rgb_output = rearrange(rgb_output, 'B C H W -> B H W C').detach().cpu().numpy()
    rgb_target = rearrange(data_retval['rgb_target'], 'B C H W -> B H W C').detach().cpu().numpy()

    # Loop over every example to obtain metrics and other info.
    all_psnr = []
    for i in range(B):
        
        # Evaluate whatever.
        mse = np.mean(np.square(rgb_output[i] - rgb_target[i]))
        psnr = -10 * np.log10(mse)
            
        # Update lists to keep.
        all_psnr.append(psnr)

    # Organize and return relevant info.
    inference_retval = dict()
    inference_retval['rgb_input'] = rgb_input  # (B, H, W, C).
    inference_retval['rgb_output'] = rgb_output  # (B, H, W, C).
    inference_retval['rgb_target'] = rgb_target  # (B, H, W, C).
    inference_retval['psnr'] = np.stack(all_psnr)  # (B).

    return inference_retval
