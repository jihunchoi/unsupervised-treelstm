from torch.autograd import Variable


def wrap_with_variable(tensor, volatile, gpu):
    if gpu > -1:
        return Variable(tensor.cuda(gpu), volatile=volatile)
    else:
        return Variable(tensor, volatile=volatile)


def unwrap_scalar_variable(var):
    if isinstance(var, Variable):
        return var.data[0]
    else:
        return var
