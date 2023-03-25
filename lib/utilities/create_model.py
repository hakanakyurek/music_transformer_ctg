from lib.model.music_transformer_ed import MusicTransformerEncoderDecoder
from lib.model.music_transformer import MusicTransformerEncoder
from lib.model.music_transformer_ctrl import MusicTransformerCTRL
from lib.model.music_transformer_cocon import  MusicTransformerCoCon
from lib.model.music_transformer_classifier import MusicTransformerClassifier
from lib.metrics.accuracy import MusicAccuracy
from lib.utilities.constants import LR_DEFAULT_START, VOCAB_SIZE_KEYS, VOCAB_SIZE_NORMAL, vocab
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
        if args.key and args.cocon:
            music_transformer = MusicTransformerEncoder(n_layers=args.n_layers, 
                                            num_heads=args.num_heads,
                                            d_model=args.d_model, 
                                            dim_feedforward=args.dim_feedforward, 
                                            dropout=args.dropout,
                                            max_sequence=args.max_sequence, 
                                            rpr=args.rpr, 
                                            acc_metric=MusicAccuracy, 
                                            loss_fn=loss_func,
                                            lr=LR_DEFAULT_START)
            
            model = MusicTransformerCoCon(music_transformer, 
                                         acc_metric=MusicAccuracy, num_heads=args.num_heads,
                                         d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                                         max_sequence=args.max_sequence, lr=args.lr, keys=args.key, loss_fn=loss_func,)
        elif args.key:
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
                                    keys=args.key)
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


def create_model_for_generation(args, model_weights):

    if args.arch == 2:
        model = MusicTransformerEncoderDecoder(n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                    max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

        model.load_state_dict(torch.load(model_weights, map_location=get_device())['state_dict'])
    elif args.arch == 1:
        if args.key:
            model = MusicTransformerCTRL(n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence=args.max_sequence, rpr=args.rpr, keys=args.key).to(get_device())
        else:
            model = MusicTransformerEncoder(n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                        max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

        model.load_state_dict(torch.load(model_weights, map_location=get_device())['state_dict'])

    return model


def create_model_for_classification(args, loss_func, n_classes):
    try:
        temp = args.key
        args.key = False
        music_transformer = create_model_for_generation(args, args.model_weights)
        args.key = temp
    except:
        args.key = False
        music_transformer = create_model_for_generation(args, args.model_weights)

    model = MusicTransformerClassifier(music_transformer=music_transformer,
                                       n_classes=n_classes,
                                       lr=args.lr, 
                                       loss_fn=loss_func)

    return model.to(get_device())


def create_model_for_classification_test(args, n_classes):
    args.dropout = 0
    args.lr = 0
    
    temp = args.key
    args.key = None
    music_transformer = create_model_for_training(args, None)
    args.key = temp
    
    model = MusicTransformerClassifier(music_transformer=music_transformer,
                                       n_classes=n_classes,
                                       lr=None, 
                                       loss_fn=None)
    
    model.load_state_dict(torch.load(args.classifier_weights, map_location=get_device())['state_dict'])

    return model.to(get_device())
