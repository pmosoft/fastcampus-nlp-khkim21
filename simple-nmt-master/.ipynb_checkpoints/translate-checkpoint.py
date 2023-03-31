import pandas as pd
import argparse
import sys
import codecs
from operator import itemgetter
import nltk

import torch

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader
from simple_nmt.models.seq2seq import Seq2Seq
from simple_nmt.models.transformer import Transformer

import execute_config as cfg
from detokenizer import detokenization_file

from subword_nmt.apply_bpe import getBpe

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        required=True,
        help='Model file name to use'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to use. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Mini batch size for parallel inference. Default=%(default)s'
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=255,
        help='Maximum sequence length for inference. Default=%(default)s'
    )
    p.add_argument(
        '--n_best',
        type=int,
        default=1,
        help='Number of best inference result per sample. Default=%(default)s'
    )
    p.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='Beam size for beam search. Default=%(default)s'
    )
    p.add_argument(
        '--lang',
        type=str,
        default=None,
        help='Source language and target language. Example: enko'
    )
    p.add_argument(
        '--length_penalty',
        type=float,
        default=1.2,
        help='Length penalty parameter that higher value produce shorter results. Default=%(default)s',
    )

    config = p.parse_args()

    return config

def read_text_from_file(config):
    lines = []

    # test.sample
    if config.realtime_sample_tf:
        fr1 = open(config.realtime_sample_fn, encoding='utf-8')
        sample_all_lines = [get_mecab_bpe(l.strip(), config) for l in fr1.read().splitlines() if l.strip()]
        sample_slice_lines = sample_all_lines[config.sample_sline:config.sample_eline]

    else:
        fr1 = open(config.test_fn + '.' + config.lang[:2], encoding='utf-8')
        sample_all_lines = [l.strip() for l in fr1.read().splitlines() if l.strip()]
        sample_slice_lines = sample_all_lines[config.sample_sline:config.sample_eline]

    fw = open(config.sample_fn, "w", encoding='utf-8')
    # sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())
    # for line in sys.stdin:
    for idx, line in enumerate(sample_slice_lines):

        if line.strip() != '':
            lines += [line.strip().split(' ')]

        if len(lines) >= config.batch_size:
            yield lines
            lines = []

        fw.write(str(idx + 1) + '|1|sample|' + line + '\n')

    fw.close()

    detokenization_file(config.sample_fn, config.sample_detoken_fn)

    # test.answer
    if not config.realtime_sample_tf:
        fr2 = open(config.test_fn + '.' + config.lang[2:4], encoding='utf-8')
        answer_all_lines = [l.strip() for l in fr2.read().splitlines() if l.strip()]
        answer_slice_lines = answer_all_lines[config.sample_sline:config.sample_eline]

        fw2 = open(config.answer_fn, "w", encoding='utf-8')
        for idx, line in enumerate(answer_slice_lines): fw2.write(str(idx + 1) + '|2|answer|' + line + '\n')
        fw2.close()

        detokenization_file(config.answer_fn, config.answer_detoken_fn)

    if len(lines) > 0:
        yield lines


def read_text(batch_size=128):
    # This method gets sentences from standard input and tokenize those.
    lines = []

    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]

        if len(lines) >= batch_size:
            yield lines
            lines = []

    if len(lines) > 0:
        yield lines


def to_text(indice, vocab):
    # This method converts index to word to show the translation result.
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                # line += ['<EOS>']
                break
            else:
                line += [vocab.itos[index]]

        line = ' '.join(line)
        lines += [line]

    return lines


def is_dsl(train_config):
    # return 'dsl_lambda' in vars(train_config).keys()
    return not ('rl_n_epochs' in vars(train_config).keys())


def get_vocabs(train_config, config, saved_data):
    if is_dsl(train_config):
        assert config.lang is not None

        if config.lang == train_config.lang:
            is_reverse = False
        else:
            is_reverse = True

        if not is_reverse:
            # Load vocabularies from the model.
            src_vocab = saved_data['src_vocab']
            tgt_vocab = saved_data['tgt_vocab']
        else:
            src_vocab = saved_data['tgt_vocab']
            tgt_vocab = saved_data['src_vocab']

        return src_vocab, tgt_vocab, is_reverse
    else:
        # Load vocabularies from the model.
        src_vocab = saved_data['src_vocab']
        tgt_vocab = saved_data['tgt_vocab']

    return src_vocab, tgt_vocab, False


def get_model(input_size, output_size, train_config, is_reverse=False):
    # Declare sequence-to-sequence model.
    if 'use_transformer' in vars(train_config).keys() and train_config.use_transformer:
        model = Transformer(
            input_size,
            train_config.hidden_size,
            output_size,
            n_splits=train_config.n_splits,
            n_enc_blocks=train_config.n_layers,
            n_dec_blocks=train_config.n_layers,
            dropout_p=train_config.dropout,
        )
    else:
        model = Seq2Seq(
            input_size,
            train_config.word_vec_size,
            train_config.hidden_size,
            output_size,
            n_layers=train_config.n_layers,
            dropout_p=train_config.dropout,
        )

    if is_dsl(train_config):
        if not is_reverse:
            model.load_state_dict(saved_data['model'][0])
        else:
            model.load_state_dict(saved_data['model'][1])
    else:
        model.load_state_dict(saved_data['model'])  # Load weight parameters from the trained model.
    model.eval()  # We need to turn-on the evaluation mode, which turns off all drop-outs.

    return model

################################################
# mecab, bpe, bleu 처리 함수
################################################
def post_token(src_line, mecab_line):
    STR = '▁'
    ref_tokens = src_line.strip().split(' ')
    input_line = mecab_line
    tokens = input_line.split(' ')

    idx = 0; buf = []
    # We assume that stdin has more tokens than reference input.
    for ref_token in ref_tokens:
        tmp_buf = []
        while idx < len(tokens):
            if tokens[idx].strip() == '':
                idx += 1
                continue
            tmp_buf += [tokens[idx]]
            idx += 1
            if ''.join(tmp_buf) == ref_token:
                break
        if len(tmp_buf) > 0:
            buf += [STR + tmp_buf[0].strip()] + tmp_buf[1:]
    return ' '.join(buf)

def get_mecab(line):
    import MeCab
    mecab = MeCab.Tagger()
    wakati = MeCab.Tagger("-Owakati")
    out = wakati.parse(line).split()
    mecab_line = ''
    for l in out: mecab_line += l + ' '
    return mecab_line

def get_mecab_bpe(line, config):
    mecab_line = get_mecab(line)
    post_mecab_line = post_token(line, mecab_line)
    bpe_line = getBpe(post_mecab_line, config)
    print(mecab_line)
    print(post_mecab_line)
    print(bpe_line)
    return bpe_line

def get_bleu(line_result, line_answer):
    list_result = line_result.split(' ')
    list_answer = line_answer.split(' ')
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([list_answer], list_result)
    return BLEUscore

if __name__ == '__main__':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    # config = define_argparser()
    from argparse import Namespace

    config = cfg.translate_config(batch_size=64, model_fn='./weight/model_064_16_1.53-4.62_1.63-5.09_pth', gpu_id=-1)
    config = Namespace(**config)

    # Load saved model.
    saved_data = torch.load(config.model_fn, map_location='cpu',)

    # Load configuration setting in training.
    train_config = saved_data['config']

    src_vocab, tgt_vocab, is_reverse = get_vocabs(train_config, config, saved_data)

    # Initialize dataloader, but we don't need to read training & test corpus.
    # What we need is just load vocabularies from the previously trained model.
    loader = DataLoader()
    loader.load_vocab(src_vocab, tgt_vocab)

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, train_config, is_reverse)

    # Put models to device if it is necessary.
    if config.gpu_id >= 0: model.cuda(config.gpu_id)

    with torch.no_grad():
        # Get sentences from standard input.

        fw = open(config.result_fn, "w", encoding='utf-8')
        for lines in read_text_from_file(config):
            # for lines in read_text(batch_size=config.batch_size):
            # Since packed_sequence must be sorted by decreasing order of length,
            # sorting by length in mini-batch should be restored by original order.
            # Therefore, we need to memorize the original index of the sentence.
            lengths = [len(line) for line in lines]
            original_indice = [i for i in range(len(lines))]

            sorted_tuples = sorted(zip(lines, lengths, original_indice), key=itemgetter(1), reverse=True,)
            sorted_lines = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
            lengths = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
            original_indice = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]

            # Converts string to list of index.
            x = loader.src.numericalize(
                loader.src.pad(sorted_lines),
                device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
            )
            # |x| = (batch_size, length)

            if config.beam_size == 1:
                y_hats, indice = model.search(x)
                # |y_hats| = (batch_size, length, output_size)
                # |indice| = (batch_size, length)

                output = to_text(indice, loader.tgt.vocab)
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    fw.write(str(i + 1) + '|3|result|' + '\n'.join(output[i]) + '\n')
                    # sys.stdout.write('\n'.join(output[i]) + '\n')

            else:
                # Take mini-batch parallelized beam search.
                batch_indice, _ = model.batch_beam_search(
                    x,
                    beam_size=config.beam_size,
                    max_length=config.max_length,
                    n_best=config.n_best,
                    length_penalty=config.length_penalty,
                )

                # Restore the original_indice.
                output = []
                for i in range(len(batch_indice)):
                    output += [to_text(batch_indice[i], loader.tgt.vocab)]
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    fw.write(str(i + 1) + '|3|result|' + '\n'.join(output[i]) + '\n')
                    # sys.stdout.write('\n'.join(output[i]) + '\n')
        fw.close()

    ################################################
    # 출력정보 생성
    ################################################
    detokenization_file(config.result_fn, config.result_detoken_fn)
    sample_df = pd.read_csv(config.sample_detoken_fn, sep='|', header=None, names=['num', 'cd', 'cd_nm', 'data'], dtype=str)
    result_df = pd.read_csv(config.result_detoken_fn, sep='|', header=None, names=['num', 'cd', 'cd_nm', 'data'], dtype=str)

    if config.realtime_sample_tf:
        unity_df = pd.concat([sample_df, result_df]).sort_values(by=['num', 'cd'])
    else:
        answer_df = pd.read_csv(config.answer_detoken_fn, sep='|', header=None, names=['num', 'cd', 'cd_nm', 'data'], dtype=str)
        unity_df = pd.concat([sample_df, answer_df, result_df]).sort_values(by=['num', 'cd'])

    print(unity_df.to_csv())

    ################################################
    # bleu score 산출
    ################################################
    line_answer=unity_df[unity_df['cd'] == '2'][["data"]].values.tolist()
    line_result=unity_df[unity_df['cd'] == '3'][["data"]].values.tolist()
    bleu_score = 0.0
    #bleu_score_list = []
    for i in range(len(line_result)):
        mecab_result = get_mecab(line_result[i][0].strip())
        mecab_answer = get_mecab(line_answer[i][0].strip())
        bleu_score += get_bleu(mecab_result, mecab_answer)
        #bleu_score_list.append(get_bleu(line_result[i][0].strip(), line_answer[i][0].strip()))
    bleu_score = bleu_score / len(line_result)
    print(f'bleu_score={bleu_score}')