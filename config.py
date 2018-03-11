data_config = {
    'word_size' : 16,
    'word_count_threshold' : 10,
    'char_count_threshold' : 50,
    'pickle_file' : 'vocabs.pkl',
}

model_config = {
    'hidden_dim'     	: 50,
    'num_layers'        : 2,
    'attention_dim'     : 2,
    'char_convs'     	: 50,
    'char_emb_dim'   	: 8,
    'dropout'        	: 0.2,
    'highway_layers' 	: 2,
    'two_step'          : True,
    'use_cudnn'         : True,
    'pointer_importance': 2,

}

training_config = {
    'minibatch_size'    : 128,    # in samples when using ctf reader, per worker
   # 'epoch_size'        : 2,   # in sequences, when using ctf reader
    'epoch_size'        : 44961,   # in sequences, when using ctf reader
    'log_freq'          : 100,     # in minibatchs
    'max_epochs'        : 300,
    'lr'                : 0.005,
    'train_data'        : 'train.ctf',  # or 'train.tsv'
    'val_data'          : 'dev.ctf',
    'val_interval'      : 5,       # interval in epochs to run validation
    'stop_after'        : 100,       # num epochs to stop if no CV improvement
    'minibatch_seqs'    : 16,      # num sequences of minibatch, when using tsv reader, per worker
    'distributed_after' : 0,       # num sequences after which to start distributed training
}
