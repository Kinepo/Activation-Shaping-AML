target_domain=${1}
experiment_choose=${2}
name_folder=${3}
list_layers_choose=${4}
hyper_parameter_choose=${5}
random_parameter_choose=${6}
activate_evalforDA_choose=${7}


python main.py \
--experiment=${experiment_choose} \
--experiment_name=${experiment_choose}/${name_folder}/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1 \
--list_layers=${list_layers_choose} \
--hyper_parameter=${hyper_parameter_choose} \
--random_parameter=${random_parameter_choose} \
--activate_evalforDA=${activate_evalforDA_choose}