checkpoints_path=$1
configs=$2
gpu=${3:-0}
metrics=${4:-"normal"}
subset=${5:-"test"}

output_path=$(dirname -- "$checkpoints_path")
out_name=$(basename -- "$configs")

mkdir -p $output_path/exp

CUDA_VISIBLE_DEVICES=$gpu python generate.py \
        --data ../data/binary/wmt14_en_de_joined_dict  \
        --path "$checkpoints_path" \
        --gen-subset $subset \
        --beam 4 \
        --batch-size 64 \
        --remove-bpe \
        --lenpen 0.6 \
        --configs=$configs \
        > $output_path/exp/${out_name}_${gpu}_gen.out

GEN=$output_path/exp/${out_name}_${gpu}_gen.out

SYS=$GEN.sys
REF=$GEN.ref

# get normal BLEU or SacreBLEU score
if [ $metrics = "normal" ]
then
  echo "Evaluate Normal BLEU score!"
  grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
  grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
  python tools/score.py --sys $SYS --ref $REF > "$(dirname -- "$checkpoints_path")/score.log"
elif [ $metrics = "sacre" ]
then
  echo "Evaluate SacreBLEU score!"
  grep ^H $GEN | cut -f3- > $SYS.pre
  grep ^T $GEN | cut -f2- > $REF.pre
  sacremoses detokenize < $SYS.pre > $SYS
  sacremoses detokenize < $REF.pre > $REF
  python tools/score.py --sys $SYS --ref $REF --sacrebleu > "$(dirname -- "$checkpoints_path")/score.log"
fi

cat "$(dirname -- "$checkpoints_path")/score.log"