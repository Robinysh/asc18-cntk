import cntk as C
import numpy as np
from polymath import PolyMath
from squad_utils import metric_max_over_ground_truths, f1_score, exact_match_score
from itertools import groupby
from operator import itemgetter
import tsv2ctf
import os
import argparse
import importlib
import time
import json
import pickle
from nltk import ngrams

import rouge

model_name = "pm.model"

#from cntk.internal.utils import _py_dict_to_cntk_dict

#def get_checkpoint_state(self):
#    return _py_dict_to_cntk_dict({})

#def restore_from_checkpoint(self, checkpoint):
 #   pass

def argument_by_name(func, name):
    found = [arg for arg in func.arguments if arg.name == name]


    if len(found) == 0:
        raise ValueError('no matching names in arguments')
    elif len(found) > 1:
        raise ValueError('multiple matching names in arguments')
    else:
        return found[0]

def create_mb_and_map(func, data_file, polymath, randomize=True, repeat=True):
    mb_source = C.io.MinibatchSource(
        C.io.CTFDeserializer(
            data_file,
            C.io.StreamDefs(
                context_g_words  = C.io.StreamDef('cgw', shape=polymath.wg_dim,     is_sparse=True),
                query_g_words    = C.io.StreamDef('qgw', shape=polymath.wg_dim,     is_sparse=True),
                context_ng_words = C.io.StreamDef('cnw', shape=polymath.wn_dim,     is_sparse=True),
                query_ng_words   = C.io.StreamDef('qnw', shape=polymath.wn_dim,     is_sparse=True),
                context_words    = C.io.StreamDef('cw',  shape=polymath.vocab_size, is_sparse=True),
                answer_words     = C.io.StreamDef('aw',  shape=polymath.vocab_size + 1,     is_sparse=True),
                answer_begin     = C.io.StreamDef('ab',  shape=polymath.a_dim,      is_sparse=False),
                answer_end       = C.io.StreamDef('ae',  shape=polymath.a_dim,      is_sparse=False),
                context_chars    = C.io.StreamDef('cc',  shape=polymath.word_size,  is_sparse=False),
                query_chars      = C.io.StreamDef('qc',  shape=polymath.word_size,  is_sparse=False))),
        randomize=randomize,
        max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)

    input_map = {
        argument_by_name(func, 'cgw'): mb_source.streams.context_g_words,
        argument_by_name(func, 'qgw'): mb_source.streams.query_g_words,
        argument_by_name(func, 'cnw'): mb_source.streams.context_ng_words,
        argument_by_name(func, 'qnw'): mb_source.streams.query_ng_words,
        argument_by_name(func, 'aw' ): mb_source.streams.answer_words,
        argument_by_name(func, 'cc' ): mb_source.streams.context_chars,
        argument_by_name(func, 'qc' ): mb_source.streams.query_chars,
        argument_by_name(func, 'ab' ): mb_source.streams.answer_begin,
        argument_by_name(func, 'ae' ): mb_source.streams.answer_end,
        argument_by_name(func, 'cw' ): mb_source.streams.context_words
      
    }
    return mb_source, input_map

def create_tsv_reader(func, tsv_file, polymath, seqs, num_workers, is_test=False, misc=None):
    with open(tsv_file, 'r', encoding='utf-8') as f:
        eof = False
        batch_count = 0
        while not(eof and (batch_count % num_workers) == 0):
            batch_count += 1
            batch={'cwids':[], 'qwids':[], 'awids':[], 'baidx':[], 'eaidx':[], 'ccids':[], 'qcids':[]}

            while not eof and len(batch['cwids']) < seqs:
                line = f.readline()
                if not line:
                    eof = True
                    break

                if misc is not None:
                    import re
                    misc['uid'].append(re.match('^([^\t]*)', line).groups()[0])

                ctokens, qtokens, atokens, cwids, qwids, awids, baidx, eaidx, ccids, qcids  = tsv2ctf.tsv_iter(line, polymath.vocab, polymath.chars, is_test, misc)

                batch['cwids'].append(cwids)
                batch['qwids'].append(qwids)
                batch['awids'].append(awids)
                batch['baidx'].append(baidx)
                batch['eaidx'].append(eaidx)
                batch['ccids'].append(ccids)
                batch['qcids'].append(qcids)
#                batch['acids'].append(acids)

            if len(batch['cwids']) > 0:
                context_g_words  = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in cwids] for cwids in batch['cwids']], polymath.wg_dim)
                context_words  = C.Value.one_hot([[i for i in cwids] for cwids in batch['cwids']], polymath.vocab_size)
                context_ng_words = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in cwids] for cwids in batch['cwids']], polymath.wn_dim)
                query_g_words    = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in qwids] for qwids in batch['qwids']], polymath.wg_dim)
                query_ng_words   = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in qwids] for qwids in batch['qwids']], polymath.wn_dim)
                answer_words   = C.Value.one_hot([[i for i in awids] for awids in batch['awids']], polymath.vocab_size+1)
                context_chars = [np.asarray([[[c for c in cc+[0]*max(0,polymath.word_size-len(cc))]] for cc in ccid], dtype=np.float32) for ccid in batch['ccids']]
                query_chars   = [np.asarray([[[c for c in qc+[0]*max(0,polymath.word_size-len(qc))]] for qc in qcid], dtype=np.float32) for qcid in batch['qcids']]
                answer_begin = [np.asarray(ab, dtype=np.float32) for ab in batch['baidx']]
                answer_end   = [np.asarray(ae, dtype=np.float32) for ae in batch['eaidx']]

                yield { argument_by_name(func, 'cgw'): context_g_words,
                        argument_by_name(func, 'qgw'): query_g_words,
                        argument_by_name(func, 'cnw'): context_ng_words,
                        argument_by_name(func, 'qnw'): query_ng_words,
                        argument_by_name(func, 'cw' ): context_words,
                        argument_by_name(func, 'aw' ): answer_words,
                        argument_by_name(func, 'cc' ): context_chars,
                        argument_by_name(func, 'qc' ): query_chars,
       #                 argument_by_name(func, 'ab' ): answer_begin,
      #                  argument_by_name(func, 'ae' ): answer_end 
                       }
            else:
                yield {} # need to generate empty batch for distributed training

def train(i2w, data_path, model_path, log_file, config_file, restore=True, profiling=False, gen_heartbeat=False):
    polymath = PolyMath(config_file)
    z, loss = polymath.model()
    training_config = importlib.import_module(config_file).training_config
    max_epochs = training_config['max_epochs']
    log_freq = training_config['log_freq']

    progress_writers = [C.logging.ProgressPrinter(
                            num_epochs = max_epochs,
                            freq = log_freq,
                            tag = 'Training',
                            log_to_file = log_file,
                            metric_is_pct=False,
                            rank = C.Communicator.rank(),
                            gen_heartbeat = gen_heartbeat)]

    lr = C.learning_parameter_schedule(training_config['lr'], minibatch_size=None, epoch_size=None)

    ema = {}
    dummies = []
    for p in z.parameters:
        ema_p = C.constant(0, shape=p.shape, dtype=p.dtype, name='ema_%s' % p.uid)
        ema[p.uid] = ema_p
        dummies.append(C.reduce_sum(C.assign(ema_p, 0.999 * ema_p + 0.001 * p)))
    dummy = C.combine(dummies)

#    learner = C.adadelta(z.parameters, lr)
    learner = C.fsadagrad(z.parameters,
                          #apply the learning rate as if it is a minibatch of size 1
                          lr ,
                          momentum = C.momentum_schedule(0.9366416204111472,minibatch_size = training_config['minibatch_size']),
                          gradient_clipping_threshold_per_sample=2.3,
                          gradient_clipping_with_truncation=True)
    if C.Communicator.num_workers() > 1:
        learner = C.data_parallel_distributed_learner(learner)

    trainer = C.Trainer(z, loss , learner, progress_writers)

    if profiling:
        C.debugging.start_profiler(sync_gpu=True)

    train_data_file = os.path.join(data_path, training_config['train_data'])
    train_data_ext = os.path.splitext(train_data_file)[-1].lower()

    model_file = os.path.join(model_path, model_name)
    model = C.combine(z.outputs + loss.outputs) #this is for validation only


    epoch_stat = {
        'best_val_err' : 1000,
        'best_since'   : 0,
        'val_since'    : 0}
    print(restore,os.path.isfile(model_file))
#    if restore and os.path.isfile(model_file):
    if restore and os.path.isfile(model_file):
        z.restore(model_file)
        #after restore always re-evaluate
        #TODO replace with rougel with external script(possibly)
        #epoch_stat['best_val_err'] = validate_model(i2w, os.path.join(data_path, training_config['val_data']), model, polymath)

    def post_epoch_work(epoch_stat):
        trainer.summarize_training_progress()
        epoch_stat['val_since'] += 1

        if epoch_stat['val_since'] == training_config['val_interval']:
            epoch_stat['val_since'] = 0
            temp = dict((p.uid, p.value) for p in z.parameters)
            for p in trainer.model.parameters:
                p.value = ema[p.uid].value
            #TODO replace with rougel with external script(possibly)
            val_err = validate_model(i2w, os.path.join(data_path, training_config['val_data']), model, polymath)
            #if epoch_stat['best_val_err'] > val_err:
            #    epoch_stat['best_val_err'] = val_err
            #    epoch_stat['best_since'] = 0
            #    trainer.save_checkpoint(model_file)
            #    for p in trainer.model.parameters:
            #        p.value = temp[p.uid]
            #else:
            #    epoch_stat['best_since'] += 1
            #    if epoch_stat['best_since'] > training_config['stop_after']:
            #        return False
            z.save(model_file)
            epoch_stat['best_since'] += 1
            if epoch_stat['best_since'] > training_config['stop_after']:
               return False
        if profiling:
            C.debugging.enable_profiler()

        return True

    init_pointer_importance = polymath.pointer_importance
    if train_data_ext == '.ctf':
        mb_source, input_map = create_mb_and_map(loss, train_data_file, polymath)

        minibatch_size = training_config['minibatch_size'] # number of samples
        epoch_size = training_config['epoch_size']

        for epoch in range(max_epochs):
            num_seq = 0
            while True:
                if trainer.total_number_of_samples_seen >= training_config['distributed_after']:
                    data = mb_source.next_minibatch(minibatch_size*C.Communicator.num_workers(), input_map=input_map, num_data_partitions=C.Communicator.num_workers(), partition_index=C.Communicator.rank())
                else:
                    data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
                trainer.train_minibatch(data)
                num_seq += trainer.previous_minibatch_sample_count
                dummy.eval()
                if num_seq >= epoch_size:
                    break
            if not post_epoch_work(epoch_stat):
                break
            print('Before Pointer_importance:', polymath.pointer_importance)
            if polymath.pointer_importance > 0.1*init_pointer_importance:
                polymath.pointer_importance =  polymath.pointer_importance * 0.9 
                print('Pointer_importance:', polymath.pointer_importance)
    else:
        if train_data_ext != '.tsv':
            raise Exception("Unsupported format")

        minibatch_seqs = training_config['minibatch_seqs'] # number of sequences

        for epoch in range(max_epochs):       # loop over epochs
            tsv_reader = create_tsv_reader(loss, train_data_file, polymath, minibatch_seqs, C.Communicator.num_workers())
            minibatch_count = 0
            for data in tsv_reader:
                if (minibatch_count % C.Communicator.num_workers()) == C.Communicator.rank():
                    trainer.train_minibatch(data) # update model with it
                    dummy.eval()
                minibatch_count += 1
            if not post_epoch_work(epoch_stat):
                break

    if profiling:
        C.debugging.stop_profiler()

def symbolic_best_span(begin, end):
    running_max_begin = C.layers.Recurrence(C.element_max, initial_state=-float("inf"))(begin)
    return C.layers.Fold(C.element_max, initial_state=C.constant(-1e+30))(running_max_begin + end)

def validate_model(i2w, test_data, model, polymath):
    print('validating')
    RL = rouge.Rouge()
    testout = model.outputs[1]  # according to model.shape
    start_logits = model.outputs[2]  
    end_logits = model.outputs[3]   
    context = model.outputs[4]
    loss = model.outputs[5]
    root = C.as_composite(loss.owner)
    mb_source, input_map = create_mb_and_map(root, test_data, polymath, randomize=False, repeat=False)
    begin_label = argument_by_name(root, 'ab')
    end_label   = argument_by_name(root, 'ae')
    onehot = argument_by_name(root, 'aw')
  



    begin_prediction = C.sequence.input_variable(1, sequence_axis=begin_label.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(1, sequence_axis=end_label.dynamic_axes[1], needs_gradient=True) 
    predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))
    true_span = C.layers.Recurrence(C.plus)(begin_label - C.sequence.past_value(end_label))  

    
    best_span_score = symbolic_best_span(begin_prediction, end_prediction)

    
    one2num = C.argmax(onehot,0)

    minibatch_size = 128
    num_sequences = 0

    stat = np.array([0,0,0,0,0,0], dtype = np.dtype('float64'))
    loss_sum = 0
    cnt = 0
    #while True:
    while cnt<1000:
        data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
        if not data or not (onehot in data) or data[onehot].num_sequences == 0:
            break

        out = model.eval(data, outputs=[testout,start_logits,end_logits, context,loss], as_numpy=True)
        true = one2num.eval({onehot:data[onehot]})
 

        g = best_span_score.grad({begin_prediction:out[start_logits], end_prediction:out[end_logits]}, wrt=[begin_prediction,end_prediction], as_numpy=False)
#        print(g[begin_prediction], g[end_prediction])
        other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction]}
        span = predicted_span.eval((other_input_map))
#        print(span)




        span_out = np.asarray(span).reshape(-1).tolist()
        context_o = np.asarray(out[context]).reshape(-1).tolist()
        predict_answer = []
        for i in range(len(span_out)):
            if(span_out[i]==1):
                predict_answer.append(context_o[i])

 #       pred_out = np.asarray(out[context]).reshape(-1).tolist()
 #       predict_answer = pred_out[span_begin:span_end+1]
        if cnt < 10:
 
            #print(predict_answer)
            print(format_true_sequences(predict_answer,i2w,polymath))
            print('\n')
        cnt+=1
        true_text = format_true_sequences(np.asarray(true).reshape(-1).tolist(),i2w, polymath)
        predout_text = format_predict_sequences(np.asarray(out[testout]).reshape(-1), predict_answer , i2w, polymath)
        testloss = out[loss]
        stat += RL.calc_score(predout_text, true_text)

        loss_sum += np.sum(np.asarray(testloss))
        num_sequences += data[onehot].num_sequences


    loss_avg = loss_sum / num_sequences
    stat_avg = stat / float(num_sequences)
    print("Validated {} sequences, loss {:.4f}, RouL {:.4f}, LCS {:.4f}, LengCan {:.4f}, LenRef {:.4f}, prec {:.4f}, rec {:.4f}".format(
            num_sequences,
            loss_avg, stat_avg[0], stat_avg[1], stat_avg[2], stat_avg[3], stat_avg[4], stat_avg[5]))

    return loss_avg


def get_vocab(path):
    # get the vocab for printing output sequences in plaintext
#    vocab = [w.strip() for w in open(path).readlines()]
#    i2w = { i:w for i,w in enumerate(vocab) }
#    w2i = { w:i for i,w in enumerate(vocab) }
    with open(path, 'rb') as vf:
           known, vocab, chars = pickle.load(vf)
    i2w = { i:w for i,w in enumerate(vocab) }
    return i2w

def unique_justseen(sentence):
    #removes adjacent duplicates
    #[1,2,3,1,2,3] -> [1,2,3]
    for n in range(1,min(7, len(sentence))):
        grams = list(ngrams(sentence, n))
        i = len(grams)-n
        while i > 0:
            if grams[i] == grams[i-n]:
                sentence = np.delete(sentence, range(i, i+n))
                i-= n-1
            i-=1
    return sentence

def format_true_sequences(sequences, i2w, polymath):
    out =  [] 
    for w in unique_justseen(sequences): 
        if w < polymath.wg_dim and w != polymath.sentence_end_index:
            out.append(i2w[w])
    return " ".join(out)

def format_predict_sequences(sequences,span_ans , i2w, polymath):
    out =  []
    uni_seq = unique_justseen(sequences)
    if len(uni_seq)==1 and uni_seq[0]==polymath.unk_index:
        uni_seq=span_ans
    for w in uni_seq:
        if w < polymath.wg_dim and w != polymath.sentence_end_index:
            out.append(i2w[w])
    return " ".join(out)

def format_output_sequences(sequences,span_ans , i2w, polymath):
    out =  []
    uni_seq = unique_justseen(sequences)
    if len(uni_seq)==1 and uni_seq[0]==polymath.unk_index:
        return  span_ans
    for w in uni_seq:
        if w < polymath.wg_dim and w != polymath.sentence_end_index:
            out.append(i2w[w])
    return " ".join(out)


# map from token to char offset
def w2c_map(s, words):
    w2c=[]
    rem=s
    offset=0
    for i,w in enumerate(words):
        cidx=rem.find(w)
        assert(cidx>=0)
        w2c.append(cidx+offset)
        offset+=cidx + len(w)
        rem=rem[cidx + len(w):]
    return w2c

# get phrase from string based on tokens and their offsets
def get_answer(raw_text, tokens, start, end):
    try:
        w2c=w2c_map(raw_text, tokens)
        return raw_text[w2c[start]:w2c[end]+len(tokens[end])]
    except:
        import pdb
        pdb.set_trace()




def test(i2w ,test_data, model_path, model_file, config_file):
    #C.try_set_default_device(C.cpu())
    polymath = PolyMath(config_file)
    print(test_data, model_path, model_file, model_name)
    print(os.path.join(model_path, model_file))
    model = C.Function.load(os.path.join(model_path, model_file if model_file else model_name))
    print(model)
    output       = model.outputs[1]
#    loss         = model.outputs[5]
    start_logits = model.outputs[2]
    end_logits = model.outputs[3]
    context = model.outputs[4]
  #  loss = model.outputs[5]
    root = C.as_composite(output.owner)

    begin_prediction = C.sequence.input_variable(1, sequence_axis=start_logits.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(1, sequence_axis=end_logits.dynamic_axes[1], needs_gradient=True)
    predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))
 
    best_span_score = symbolic_best_span(begin_prediction, end_prediction)


    batch_size = 1 # in sequences
    misc = {'rawctx':[], 'ctoken':[], 'answer':[], 'uid':[]}
    tsv_reader = create_tsv_reader(root, test_data, polymath, batch_size, 1, is_test=True, misc=misc)
    results = {}
    with open('{}_out.json'.format(model_file), 'w', encoding='utf-8') as json_output:
        for data in tsv_reader:
            out = model.eval(data, outputs=[output, start_logits,end_logits, context], as_numpy=False)
            g = best_span_score.grad({begin_prediction:out[start_logits], end_prediction:out[end_logits]}, wrt=[begin_prediction,end_prediction], as_numpy=False)           
            other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction]}
            span = predicted_span.eval((other_input_map))
            for seq, (raw_text, ctokens, answer, uid) in enumerate(zip(misc['rawctx'], misc['ctoken'], misc['answer'], misc['uid'])):
     #           g = best_span_score.grad({begin_prediction:out[start_logits], end_prediction:out[end_logits]}, wrt=[begin_prediction,end_prediction], as_numpy=False)

      #          other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction]}
       #         span = predicted_span.eval((other_input_map))
                seq_where = np.argwhere(span[seq])[:,0]
                span_begin = np.min(seq_where)
                span_end = np.max(seq_where)
                predict_answer = get_answer(raw_text, ctokens, span_begin, span_end)
         #       span_out = np.asarray(span).reshape(-1).tolist()
         #       context_o = np.asarray(out[context]).reshape(-1).tolist()
         #       predict_answer = []
         #       for i in range(len(span_out)):
         #           if(span_out[i]==1):
         #               predict_answer.append(context_o[i])
                print(predict_answer)
                final_answer = format_output_sequences(np.asarray(out[output].as_sequences()).reshape(-1),predict_answer, i2w, polymath)
                results['query_id'] = int(uid)
                results['answers'] = [final_answer]
                print(results)
                json.dump(results, json_output)
                json_output.write("\n")
            misc['rawctx'] = []
            misc['ctoken'] = []
            misc['answer'] = []
            misc['uid'] = []

if __name__=='__main__':
    # default Paths relative to current python file.
    abs_path   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(abs_path, 'Models')
    data_path  = os.path.join(abs_path, '.')

    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=model_path)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
    parser.add_argument('-genheartbeat', '--genheartbeat', help="Turn on heart-beat for philly", action='store_true', default=False)
    parser.add_argument('-config', '--config', help='Config file', required=False, default='config')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_false')
    parser.add_argument('-test', '--test', help='Test data file', required=False, default=None)
    parser.add_argument('-model', '--model', help='Model file name', required=False, default=model_name)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir']
    if args['datadir'] is not None:
        data_path = args['datadir']
        
    #C.try_set_default_device(C.cpu())
    #C.try_set_default_device(C.gpu(C.Communicator.rank() % 4))
   # C.try_set_default_device(C.gpu(0))

    test_data = args['test']
    test_model = args['model']
    
    print(args['restart'])
    pickle_file = os.path.join(abs_path, 'vocabs.pkl')
    i2w  = get_vocab(pickle_file)
#    print(i2w)
    if test_data:
        test(i2w, test_data, model_path, test_model, args['config'])
    else:
        try:
            train(i2w, data_path, model_path, args['logdir'], args['config'],
                restore = not args['restart'],
                profiling = args['profile'],
                gen_heartbeat = args['genheartbeat'])
        finally:
            C.Communicator.finalize()
