target_domain=${1}
experiment_choose=${2}
name_folder=${3}

python main.py \
--experiment=${experiment_choose} \
--experiment_name=${experiment_choose}/${name_folder}/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1