import os

root_data_dir = '../'

dataset = '../../Dataset/movielens'
behaviors = 'movielens_users.txt'
news = 'movielens_items.txt'
logging_num = 4
testing_num = 1

bert_model_load = 'bert-base-uncased'
freeze_paras_before = 0
news_attributes = 'title'

mode = 'train'
item_tower = 'modal'

epoch = 60
load_ckpt_name = 'None'

l2_weight_list = [0]
drop_rate_list = [0.1]
batch_size_list = [32]
lr_list = [1e-4]
embedding_dim_list = [64]
fine_tune_lr_list = [5e-5]

for l2_weight in l2_weight_list:
    for batch_size in batch_size_list:
        for lr in lr_list:
            for embedding_dim in embedding_dim_list:
                for drop_rate in drop_rate_list:
                    for fine_tune_lr in fine_tune_lr_list:
                        label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_Flr{}'.format(
                            item_tower, batch_size, embedding_dim, lr,
                            drop_rate, l2_weight, fine_tune_lr)
                        run_py = "torchrun --nproc_per_node 1 --master_port 1238\
                                 ../run.py --root_data_dir {}  --dataset {} --behaviors {} --news {}\
                                 --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                 --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {} \
                                 --news_attributes {} --bert_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr {}".format(
                            root_data_dir, dataset, behaviors, news,
                            mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                            l2_weight, drop_rate, batch_size, lr, embedding_dim,
                            news_attributes, bert_model_load, epoch, freeze_paras_before, fine_tune_lr)
                        os.system(run_py)
