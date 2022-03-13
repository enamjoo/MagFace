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

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/lfw_turbulence_1/img.list \
                    --feat_list ${FEAT_PATH}/lfw_turbulence_1_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/lfw_turbulence_2/img.list \
                    --feat_list ${FEAT_PATH}/lfw_turbulence_2_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/lfw_turbulence_3/img.list \
                    --feat_list ${FEAT_PATH}/lfw_turbulence_3_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/lfw_turbulence_4/img.list \
                    --feat_list ${FEAT_PATH}/lfw_turbulence_4_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/lfw_turbulence_5/img.list \
                    --feat_list ${FEAT_PATH}/lfw_turbulence_5_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}



python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/agedb/img.list \
                    --feat_list ${FEAT_PATH}/agedb_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/agedb_turbulence_1/img.list \
                    --feat_list ${FEAT_PATH}/agedb_turbulence_1_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/agedb_turbulence_2/img.list \
                    --feat_list ${FEAT_PATH}/agedb_turbulence_2_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/agedb_turbulence_3/img.list \
                    --feat_list ${FEAT_PATH}/agedb_turbulence_3_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/agedb_turbulence_4/img.list \
                    --feat_list ${FEAT_PATH}/agedb_turbulence_4_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}

python3 ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/agedb_turbulence_5/img.list \
                    --feat_list ${FEAT_PATH}/agedb_turbulence_5_${FEAT_SUFFIX}.list \
                    --batch_size 256 \
                    --resume ${CKPT}



echo evaluate lfw
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_${FEAT_SUFFIX}.list \
                --pair-list data/lfw/pair.list \

echo evaluate lfw_turbulence_1
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_turbulence_1_${FEAT_SUFFIX}.list \
                --pair-list data/lfw_turbulence_1/pair.list \

echo evaluate lfw_turbulence_2
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_turbulence_2_${FEAT_SUFFIX}.list \
                --pair-list data/lfw_turbulence_2/pair.list \

echo evaluate lfw_turbulence_3
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_turbulence_3_${FEAT_SUFFIX}.list \
                --pair-list data/lfw_turbulence_3/pair.list \

echo evaluate lfw_turbulence_4
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_turbulence_4_${FEAT_SUFFIX}.list \
                --pair-list data/lfw_turbulence_4/pair.list \

echo evaluate lfw_turbulence_5
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/lfw_turbulence_5_${FEAT_SUFFIX}.list \
                --pair-list data/lfw_turbulence_5/pair.list \



echo evaluate agedb
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_${FEAT_SUFFIX}.list \
                --pair-list data/agedb/pair.list \

echo evaluate agedb_turbulence_1
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_turbulence_1_${FEAT_SUFFIX}.list \
                --pair-list data/agedb_turbulence_1/pair.list \

echo evaluate agedb_turbulence_2
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_turbulence_2_${FEAT_SUFFIX}.list \
                --pair-list data/agedb_turbulence_2/pair.list \

echo evaluate agedb_turbulence_3
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_turbulence_3_${FEAT_SUFFIX}.list \
                --pair-list data/agedb_turbulence_3/pair.list \

echo evaluate agedb_turbulence_4
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_turbulence_4_${FEAT_SUFFIX}.list \
                --pair-list data/agedb_turbulence_4/pair.list \

echo evaluate agedb_turbulence_5
python3 eval_1v1.py \
        --feat-list ${FEAT_PATH}/agedb_turbulence_5_${FEAT_SUFFIX}.list \
                --pair-list data/agedb_turbulence_5/pair.list \

