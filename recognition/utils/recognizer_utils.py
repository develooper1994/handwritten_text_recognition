import mxnet as mx

try:
    from recognition.ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding
    from recognition.ocr.utils.beam_search import ctcBeamSearch
    from recognition.ocr.utils.iam_dataset import IAMDataset
except:
    try:
        from handwritten_text_recognition.recognition.ocr.handwriting_line_recognition import \
            decode as decoder_handwriting, alphabet_encoding
        from handwritten_text_recognition.recognition.ocr.utils.beam_search import ctcBeamSearch
        from handwritten_text_recognition.recognition.ocr.utils.iam_dataset import IAMDataset
    except:
        from recognition.handwritten_text_recognition.recognition.ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding
        from recognition.handwritten_text_recognition.recognition.ocr.utils.beam_search import ctcBeamSearch
        from recognition.handwritten_text_recognition.recognition.ocr.utils.iam_dataset import IAMDataset

def device_selection_helper(device=None, num_device=1):
    """
    Helps to select possible devices.
    :param device:
    If it is None:
        If num_device==1: uses gpu if there is any gpu
        else: uses num_device gpu if there is any gpu
    If it is 'auto':
        If num_device==1: uses gpu if there is any gpu
        else: uses num_device gpu if there is any gpu
    if it is 'cpu': uses one, num_device-1 indexed cpu
    if it is 'gpu': uses one, num_device-1 indexed gpu
    :param num_device: number of device that module running on.
    :return: possible devices
    """
    num_device = abs(num_device)
    assert num_device != 0, "Please enter bigger than 1"
    if device is None:
        if num_device == 1:
            device_object = mx.gpu(0)
        else:
            device_object = [mx.gpu(i) for i in range(num_device)]
    elif device == 'auto':
        if num_device == 1:
            device_object = mx.gpu(num_device - 1) if mx.context.num_gpus() > 0 else mx.cpu(num_device - 1)
        else:
            device_object = [mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu() for i in range(num_device)]
            # device = [mx.gpu(i) for i in range(num_device)] if mx.context.num_gpus() > 0 else [mx.cpu(i) for i in
            #                                                                                    range(num_device)]
    elif device == 'cpu':
        device_object = mx.cpu(num_device - 1)
    elif device == 'gpu':
        device_object = mx.gpu(num_device - 1)
    else:
        # If it isn't a string.
        print("Assuming device is a mxnet ctx or device object or queue. Exp: mx.gpu(0)")
        device_object = device

    return device_object


## Character Probalities to Text
def get_arg_max(prob):
    """
    The greedy algorithm convert the output of the handwriting recognition network into strings.
    :param prob: probability values
    :return: maximum probabilities
    """
    arg_max = mx.nd.array(prob).topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]


def get_beam_search(prob, width=5):
    """
    Helps to get beam search probabilities
    :param prob: Probabilities
    :param width: Beam witdh
    :return: beam search probabilities
    """
    possibilities = ctcBeamSearch(prob.softmax()[0].asnumpy(), alphabet_encoding, None, width)
    return possibilities[0]


## recognizer Class to handle all mess
def get_IAMDataset_test(credentials):
    """
    Helps to get IAM dataset
    :param credentials: Account information to access IAM dataset.
    If you don't have you can have from http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php
    Register and write your account information to credentials.json
    :return: iam dataset iterator
    """
    test_ds = IAMDataset("form_original", credentials=credentials, train=False)
    return test_ds


def singleton(cls):
    instances = {}
    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()