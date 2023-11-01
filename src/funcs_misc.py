import random
import numpy as np
from prettytable import PrettyTable
from torch import jit, from_numpy

# Import network architecture from models.py
from models import BeatTrackerModel

def fullprint(*args, **kwargs) -> None:
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)

def count_parameters(model) -> int:
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params

    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def print_network_architecture(dataset,
                               model_layers,
                               show_width_expansion=False, 
                               show_parameters=False,
                               export_graph=False, 
                               seed=None,
                               device=None,
                               ) -> None:
    if seed:
        random.seed(seed)
    rand = random.randint(0, len(dataset))
    
    print('Single file structure:\n')
    example_track = dataset[rand]
    x, beats, downbeats, _ = example_track
    print('Spectrogram shape:\n', x.shape, type(x))
    print('Beats:\n', beats.shape, type(beats))
    print('Downbeats:\n', downbeats.shape, type(downbeats))

    if show_width_expansion:
        print('Sample from original beat array info: ')
        print('Total length array: ', beats.shape[1])
        print('Total number of non-zero indeces: ', np.count_nonzero(beats))
        fullprint(beats[0, :500])
        print(f'Sample from original downbeat array info: {downbeats}')
        print('Total length array: ', downbeats.shape[1])
        print(f'Total number of non-zero indeces: {np.count_nonzero(downbeats)}')
        fullprint(downbeats[0, :500])
        
    x_test = from_numpy(x).unsqueeze(0).to(device)
    print('Spectrogram tensor shape: ', x_test.shape)

    m_temp = BeatTrackerModel(num_layers=model_layers).to(device)
    beats, downbeats = m_temp(x_test)
    print('Model architecture: \n', m_temp.eval())
    print('Beat output tensor shape: ', beats.shape)
    print('Downbeat output tensor shape: ', downbeats.shape)

    if show_parameters:
        _ = count_parameters(m_temp)

    if export_graph:
        # Generate a torch.jit.ScriptModule
        traced_script_module = jit.trace(m_temp, x_test)

        # Save the TorchScript model
        traced_script_module.save(f'traced_beattrackermodel_layers{model_layers}.pt')

def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
