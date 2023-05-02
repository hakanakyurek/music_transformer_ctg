#! /bin/bash


TARGETLENGTH=1024
CLASSIFIERWEIGHTS=checkpoints/2nhk85a5/best.ckpt
MIDIROOT=dataset/encoded_new/random_seq/key/test

#### Vanilla ####

MODELWEIGHTS=artifacts/encoder6l_noaug_newpreprocess_model:v19/best.ckpt


# Input size 0
OUTPUTDIR=output/base/i1_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 1023 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT

# Input size 64
OUTPUTDIR=output/base/i64_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 64 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT

# Input size 128
OUTPUTDIR=output/base/i128_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 128 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT

# Input size 256
OUTPUTDIR=output/base/i256_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 256 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT

# Input size 512
OUTPUTDIR=output/base/i512_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 512 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT



#### COCON ####