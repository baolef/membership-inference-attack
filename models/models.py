import copy
from speechbrain.pretrained import SpeakerRecognition

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model


def get(model_spec):
    if model_spec['pretrain']:
        model = SpeakerRecognition.from_hparams(**model_spec['args'], run_opts={"device": "cuda"})
    else:
        pass
    return model
