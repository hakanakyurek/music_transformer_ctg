#! /bin/bash


TARGETLENGTH=1024
CLASSIFIERWEIGHTS=checkpoints/2nhk85a5/best.ckpt
MIDIROOT=dataset/encoded_new/random_seq/key/test
MODELWEIGHTS=artifacts/encoder6l_noaug_newpreprocess_cocon_at_2nd_layer_model:v0/best.ckpt

# Input size 0
OUTPUTDIR=output/cocon/i1_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 1 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT --key --cocon

# Input size 64
OUTPUTDIR=output/cocon/i64_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 64 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT --key --cocon

# Input size 128
OUTPUTDIR=output/cocon/i128_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 128 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT --key --cocon

# Input size 256
OUTPUTDIR=output/cocon/i256_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 256 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT --key --cocon

# Input size 512
OUTPUTDIR=output/cocon/i512_t$TARGETLENGTH/
mkdir -p $OUTPUTDIR
python test.py --rpr --num_prime 512 --max_sequence 1024 --arch 1 --output_dir $OUTPUTDIR --target_seq_length $TARGETLENGTH --model_weights $MODELWEIGHTS  --classifier_weights $CLASSIFIERWEIGHTS --midi_root $MIDIROOT --key --cocon
