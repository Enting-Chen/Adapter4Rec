import os

root_data_dir = '../'

dataset = '../../Dataset/douban'
behaviors = 'douban_books_users.txt'
news = 'douban_books_items.txt'
# for the pretrained model
pretrained_model_dir = '../pretrained_models'
pretrained_model_name = 'epoch-53'

logging_num = 4
testing_num = 1

bert_model_load = 'bert-base-multilingual-cased'
freeze_paras_before = 0
news_attributes = 'title'

mode = 'train'
item_tower = 'modal'

epoch = 100
load_ckpt_name = 'None'

l2_weight_list = [0]
drop_rate_list = [0.1]
batch_size_list = [32]
lr_list = [1e-4]
fine_tune_lr_list = [5e-5]
embedding_dim_list = [64]
adapter_bert_lr_list = [1.5e-4]
adapter_sasrec_lr_list = [1.5e-4]
# ['bert','sasrec_all', 'sasrec_first', 'sasrec_last', 'all','None']
adding_adapter_to_list = ['all']
# None or all
fine_tune_to_list = ['None']
# None or TRUE
finetune_layernorm = "None"
# None or True
is_serial = "True"
adapter_down_size = 16
bert_adapter_down_size = 64
adapter_type = "prompt"
for adapter_sasrec_lr in adapter_sasrec_lr_list:
    for adapter_bert_lr in adapter_bert_lr_list:
        for adding_adapter_to in adding_adapter_to_list:
            for fine_tune_to in fine_tune_to_list:
                for l2_weight in l2_weight_list:
                    for batch_size in batch_size_list:
                        for lr in lr_list:
                            for embedding_dim in embedding_dim_list:
                                for drop_rate in drop_rate_list:
                                    for fine_tune_lr in fine_tune_lr_list:
                                        label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_Flr{}_Blr{}_SASlr{}'.format(
                                            item_tower, batch_size, embedding_dim, lr,
                                            drop_rate, l2_weight, fine_tune_lr, adapter_bert_lr, adapter_bert_lr,
                                            adapter_sasrec_lr)
                                        run_py = "torchrun --nproc_per_node 1 --master_port 1277\
                                                 ../run.py --root_data_dir {}  --dataset {} --behaviors {} --news {}\
                                                 --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                                 --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {} \
                                                 --news_attributes {} --bert_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr {}\
                                                 --adapter_sasrec_lr {} --adapter_bert_lr {} --adding_adapter_to {} --fine_tune_to {}\
                                                 --pretrained_model_dir {} --pretrained_model_name {} --finetune_layernorm {} --is_serial {} --adapter_down_size {} --bert_adapter_down_size {} --adapter_type {}".format(
                                            root_data_dir, dataset, behaviors, news,
                                            mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                            l2_weight, drop_rate, batch_size, lr, embedding_dim,
                                            news_attributes, bert_model_load, epoch, freeze_paras_before, fine_tune_lr,
                                            adapter_sasrec_lr, adapter_bert_lr, adding_adapter_to
                                            , fine_tune_to, pretrained_model_dir, pretrained_model_name,
                                            finetune_layernorm, is_serial, adapter_down_size, bert_adapter_down_size,
                                            adapter_type)
                                        os.system(run_py)
