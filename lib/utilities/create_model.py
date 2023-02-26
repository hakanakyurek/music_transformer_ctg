from lib.model.music_transformer_ed import MusicTransformerEncoderDecoder
from lib.model.music_transformer import MusicTransformerEncoder
from lib.model.music_transformer_ctrl import MusicTransformerCTRL
from lib.model.music_transformer_classifier import MusicTransformerClassifier
from lib.metrics.accuracy import MusicAccuracy
from lib.utilities.constants import LR_DEFAULT_START
from lib.utilities.device import get_device

import torch


def create_model_for_training(args, loss_func):
    if args.arch == 2:
        model = MusicTransformerEncoderDecoder(n_layers=args.n_layers, 
                                                num_heads=args.num_heads,
                                                d_model=args.d_model, 
                                                dim_feedforward=args.dim_feedforward, 
                                                dropout=args.dropout,
                                                max_sequence=args.max_sequence, 
                                                rpr=args.rpr, 
                                                acc_metric=MusicAccuracy, 
                                                loss_fn=loss_func,
                                                lr=LR_DEFAULT_START)
    elif args.arch == 1:
        if args.keys:
            model = MusicTransformerCTRL(n_layers=args.n_layers, 
                                    num_heads=args.num_heads,
                                    d_model=args.d_model, 
                                    dim_feedforward=args.dim_feedforward, 
                                    dropout=args.dropout,
                                    max_sequence=args.max_sequence, 
                                    rpr=args.rpr, 
                                    acc_metric=MusicAccuracy, 
                                    loss_fn=loss_func,
                                    lr=LR_DEFAULT_START,
                                    keys=args.keys)
        else:
            model = MusicTransformerEncoder(n_layers=args.n_layers, 
                                            num_heads=args.num_heads,
                                            d_model=args.d_model, 
                                            dim_feedforward=args.dim_feedforward, 
                                            dropout=args.dropout,
                                            max_sequence=args.max_sequence, 
                                            rpr=args.rpr, 
                                            acc_metric=MusicAccuracy, 
                                            loss_fn=loss_func,
                                            lr=LR_DEFAULT_START)
    
    return model


def create_model_for_generation(args):

    if args.arch == 2:
        model = MusicTransformerEncoderDecoder(n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                    max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

        model.load_state_dict(torch.load(args.model_weights, map_location=get_device())['state_dict'])
    elif args.arch == 1:
        if args.key:
            model = MusicTransformerCTRL(n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence=args.max_sequence, rpr=args.rpr, keys=args.key).to(get_device())
        else:
            model = MusicTransformerEncoder(n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

        model.load_state_dict(torch.load(args.model_weights, map_location=get_device())['state_dict'])

    return model


def create_model_for_classification(args, loss_func):

    music_transformer = create_model_for_generation(args)

    model = MusicTransformerClassifier(music_transformer=music_transformer,
                                       n_classes=0,
                                       lr=args.lr, 
                                       loss_fn=loss_func)

    return model

