import random


def subsample_data(data, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), subsample_size)
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    return inputs, outputs

def subsample_data_for_cross_val(data, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), subsample_size)
    #print('range(len(inputs)', (len(inputs), indices))
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    input_compl = [data[0][i] for i in range(len(data[0])) if i not in indices]
    output_compl = [data[1][i] for i in range(len(data[0])) if i not in indices]
    return [inputs, outputs], [input_compl, output_compl]


def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), split_size)
    inputs1 = [inputs[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    return (inputs1, outputs1), (inputs2, outputs2)
