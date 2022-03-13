CKPT=$1
FEAT_SUFFIX=$2
NL=$3

ARCH=iresnet${NL}
FEAT_PATH=./features/magface_${ARCH}/
mkdir -p ${FEAT_PATH}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/lfw/img.list \
                    --feat_list ${FEAT_PATH}/lfw_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/lfw_blur_1/img.list \
#                     --feat_list ${FEAT_PATH}/lfw_blur_1_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/lfw_blur_2/img.list \
#                     --feat_list ${FEAT_PATH}/lfw_blur_2_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/lfw_blur_3/img.list \
#                     --feat_list ${FEAT_PATH}/lfw_blur_3_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/lfw_blur_4/img.list \
#                     --feat_list ${FEAT_PATH}/lfw_blur_4_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/lfw_blur_5/img.list \
#                     --feat_list ${FEAT_PATH}/lfw_blur_5_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}



# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/agedb/img.list \
#                     --feat_list ${FEAT_PATH}/agedb_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/agedb_blur_1/img.list \
#                     --feat_list ${FEAT_PATH}/agedb_blur_1_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/agedb_blur_2/img.list \
#                     --feat_list ${FEAT_PATH}/agedb_blur_2_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/agedb_blur_3/img.list \
#                     --feat_list ${FEAT_PATH}/agedb_blur_3_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/agedb_blur_4/img.list \
#                     --feat_list ${FEAT_PATH}/agedb_blur_4_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/agedb_blur_5/img.list \
#                     --feat_list ${FEAT_PATH}/agedb_blur_5_${FEAT_SUFFIX}.list \
#                     --batch_size 256 \
#                     --resume ${CKPT}



echo evaluate lfw
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_${FEAT_SUFFIX}.list \
                --pair-list data/lfw/pair.list \

echo evaluate lfw
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_${FEAT_SUFFIX}.list \
                --pair-list data/lfw/pair.list \
                
echo evaluate lfw_blur_1
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_blur_1_${FEAT_SUFFIX}.list \
                --pair-list data/lfw_blur_1/pair.list \

echo evaluate lfw_blur_2
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_blur_2_${FEAT_SUFFIX}.list \
                --pair-list data/lfw_blur_2/pair.list \

echo evaluate lfw_blur_3
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_blur_3_${FEAT_SUFFIX}.list \
                --pair-list data/lfw_blur_3/pair.list \

echo evaluate lfw_blur_4
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_blur_4_${FEAT_SUFFIX}.list \
                --pair-list data/lfw_blur_4/pair.list \

echo evaluate lfw_blur_5
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_blur_5_${FEAT_SUFFIX}.list \
                --pair-list data/lfw_blur_5/pair.list \



echo evaluate agedb
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_${FEAT_SUFFIX}.list \
                --pair-list data/agedb/pair.list \

echo evaluate agedb_blur_1
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_blur_1_${FEAT_SUFFIX}.list \
                --pair-list data/agedb_blur_1/pair.list \

echo evaluate agedb_blur_2
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_blur_2_${FEAT_SUFFIX}.list \
                --pair-list data/agedb_blur_2/pair.list \

echo evaluate agedb_blur_3
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_blur_3_${FEAT_SUFFIX}.list \
                --pair-list data/agedb_blur_3/pair.list \

echo evaluate agedb_blur_4
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_blur_4_${FEAT_SUFFIX}.list \
                --pair-list data/agedb_blur_4/pair.list \

echo evaluate agedb_blur_5
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_blur_5_${FEAT_SUFFIX}.list \
                --pair-list data/agedb_blur_5/pair.list \

