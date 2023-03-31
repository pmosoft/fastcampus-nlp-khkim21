def orig_config():
    return {
              'load_fn'             : True
            , 'model_fn'            : 'model.pth'
            , 'train'               : 'data/corpus.shuf.train.tok.bpe'
            , 'valid'               : 'data/corpus.shuf.valid.tok.bpe'
            , 'lang'                : 'enko'
            , 'gpu_id'              : -1
            , 'off_autocast'        : True
            , 'batch_size'          : 128
            , 'n_epochs'            : 30
            , 'verbose'             : 2
            , 'init_epoch'          : 1
            , 'max_length'          : 100
            , 'dropout'             : .2
            , 'word_vec_size'       : 512
            , 'hidden_size'         : 768
            , 'n_layers'            : 4
            , 'max_grad_norm'       : 1e+8
            , 'iteration_per_update': 2
            , 'lr'                  : 1e-3
            , 'lr_step'             : 0
            , 'use_adam'            : True
            , 'use_radam'           : True
            , 'rl_lr'               : .01
            , 'rl_n_samples'        : 1
            , 'rl_n_epochs'         : 10
            , 'rl_n_gram'           : 6
            , 'rl_reward'           : 'gleu'
            , 'use_transformer'     : False
            , 'n_splits'            : 8
    }

def init_config(batch_size,train,valid,n_epochs,gpu_id):
    return {
              'load_fn'             : True
            , 'model_fn'            : 'model.pth'
            , 'train'               : train
            , 'valid'               : valid
            , 'lang'                : 'enko'
            , 'gpu_id'              : gpu_id
            , 'off_autocast'        : False if gpu_id >= 0 else True
            , 'batch_size'          : batch_size
            , 'n_epochs'            : n_epochs
            , 'verbose'             : 2
            , 'init_epoch'          : 1
            , 'max_length'          : 100
            , 'dropout'             : .2
            , 'word_vec_size'       : 512
            , 'hidden_size'         : 768
            , 'n_layers'            : 4
            , 'max_grad_norm'       : 1e+8
            , 'iteration_per_update': 2
            , 'lr'                  : 1e-3
            , 'lr_step'             : 0
            , 'use_adam'            : True
            , 'use_radam'           : False
            , 'rl_lr'               : .01
            , 'rl_n_samples'        : 1
            , 'rl_n_epochs'         : 0
            , 'rl_n_gram'           : 6
            , 'rl_reward'           : 'gleu'
            , 'use_transformer'     : False
            , 'n_splits'            : 8
    }

def continue_config(batch_size, load_fn, init_epoch, n_epochs, gpu_id):
    return {
              'load_fn'             : load_fn
            , 'model_fn'            : 'NA'
            , 'train'               : 'NA'
            , 'valid'               : 'NA'
            , 'lang'                : 'NA'
            , 'gpu_id'              : gpu_id
            , 'off_autocast'        : False if gpu_id >= 0 else True
            , 'batch_size'          : batch_size
            , 'n_epochs'            : n_epochs
            , 'verbose'             : 'NA'
            , 'init_epoch'          : init_epoch
            , 'max_length'          : 'NA'
            , 'dropout'             : 'NA'
            , 'word_vec_size'       : 'NA'
            , 'hidden_size'         : 'NA'
            , 'n_layers'            : 'NA'
            , 'max_grad_norm'       : 'NA'
            , 'iteration_per_update': 'NA'
            , 'lr'                  : 'NA'
            , 'lr_step'             : 'NA'
            , 'use_adam'            : 'NA'
            , 'use_radam'           : 'NA'
            , 'rl_lr'               : 'NA'
            , 'rl_n_samples'        : 'NA'
            , 'rl_n_epochs'         : 'NA'
            , 'rl_n_gram'           : 'NA'
            , 'rl_reward'           : 'NA'
            , 'use_transformer'     : 'NA'
            , 'n_splits'            : 'NA'
    }

def translate_config(batch_size, model_fn, gpu_id):
    return {
              'model_fn'            : model_fn
            , 'gpu_id'              : gpu_id
            , 'batch_size'          : batch_size
            , 'max_length'          : 255
            , 'n_best'              : 1
            , 'beam_size'           : 5
            , 'lang'                : 'enko'
            , 'length_penalty'      : 1.2
            , 'sample_fn'           : './data/translate.sample.txt'
            , 'sample_sline'        : 0
            , 'sample_eline'        : 50
            , 'sample_detoken_fn'   : './data/translate.sample.detoken.txt'
            , 'answer_fn'           : './data/translate.answer.txt'
            , 'answer_detoken_fn'   : './data/translate.answer.detoken.txt'
            , 'result_fn'           : './data/translate.result.txt'
            , 'result_detoken_fn'   : './data/translate.result.detoken.txt'
    }