target_domain=${1}
experiment_choose=${2}
name_folder=${3}
num_layer_choose=${4}
num_block_choose=${5}
num_bn_choose=${6}

python main.py \
--experiment=${experiment_choose} \
--experiment_name=${experiment_choose}/${name_folder}/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1 \
--num_layer = ${num_layer_choose} \
--num_block = ${num_block_choose} \
--num_bn = ${num_bn_choose}