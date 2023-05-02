#### CTRL ####

#! /bin/bash


TARGETLENGTH=1024
CLASSIFIERWEIGHTS=checkpoints/2nhk85a5/best.ckpt
MIDIROOT=dataset/encoded_new/random_seq/key/test
MODELWEIGHTS=models/encoder6l_noaug_newpreprocess_key_transposed.ckpt

# Input size 0
OUTPUTDIR=output/ctrl/i1_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 1 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT --key

# Input size 64
OUTPUTDIR=output/ctrl/i64_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 64 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT --key

# Input size 128
OUTPUTDIR=output/ctrl/i128_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 128 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT --key

# Input size 256
OUTPUTDIR=output/ctrl/i256_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 256 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT --key

# Input size 512
OUTPUTDIR=output/ctrl/i512_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 512 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT --key
