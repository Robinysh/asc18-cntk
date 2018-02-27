import cntk as C
import numpy as np
from polymath import PolyMath
from squad_utils import metric_max_over_ground_truths, f1_score, exact_match_score
import tsv2ctf
import os
import argparse
import importlib
import time
import json
import pickle

model_name = "pm.model"

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
                answer_words     = C.io.StreamDef('aw',  shape=polymath.vocab_size + 2,     is_sparse=True),
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
        argument_by_name(func, 'qc' ): mb_source.streams.query_chars
    }
    return mb_source, input_map

def create_tsv_reader(func, tsv_file, polymath, seqs, num_workers, is_test=False, misc=None):
    with open(tsv_file, 'r', encoding='utf-8') as f:
        eof = False
        batch_count = 0
        while not(eof and (batch_count % num_workers) == 0):
            batch_count += 1
            batch={'cwids':[], 'qwids':[], 'awids':[], 'ccids':[], 'qcids':[], 'acids':[]}

            while not eof and len(batch['cwids']) < seqs:
                line = f.readline()
                if not line:
                    eof = True
                    break

                if misc is not None:
                    import re
                    misc['uid'].append(re.match('^([^\t]*)', line).groups()[0])

                ctokens, qtokens, atokens, cwids, qwids, awids, ccids, qcids, acids = tsv2ctf.tsv_iter(line, polymath.vocab, polymath.chars, is_test, misc)

                batch['cwids'].append(cwids)
                batch['qwids'].append(qwids)
                batch['awids'].append(awids)
                batch['ccids'].append(ccids)
                batch['qcids'].append(qcids)
                batch['acids'].append(acids)

            if len(batch['cwids']) > 0:
                context_g_words  = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in cwids] for cwids in batch['cwids']], polymath.wg_dim)
                context_ng_words = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in cwids] for cwids in batch['cwids']], polymath.wn_dim)
                context_ng_words = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in cwids] for cwids in batch['cwids']], polymath.wn_dim)
                query_g_words    = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in qwids] for qwids in batch['qwids']], polymath.wg_dim)
                query_ng_words   = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in qwids] for qwids in batch['qwids']], polymath.wn_dim)
                answer_g_words    = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in awids] for awids in batch['awids']], polymath.wg_dim)
                answer_ng_words   = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in awids] for awids in batch['awids']], polymath.wn_dim)
                answer_words   = C.Value.one_hot([[i for i in awids] for awids in batch['awids']], polymath.vocab_size+2)
                context_chars = [np.asarray([[[c for c in cc+[0]*max(0,polymath.word_size-len(cc))]] for cc in ccid], dtype=np.float32) for ccid in batch['ccids']]
                query_chars   = [np.asarray([[[c for c in qc+[0]*max(0,polymath.word_size-len(qc))]] for qc in qcid], dtype=np.float32) for qcid in batch['qcids']]
                answer_chars   = [np.asarray([[[c for c in ac+[0]*max(0,polymath.word_size-len(ac))]] for ac in acid], dtype=np.float32) for acid in batch['acids']]

                yield { argument_by_name(func, 'cgw'): context_g_words,
                        argument_by_name(func, 'qgw'): query_g_words,
                        argument_by_name(func, 'agw'): answer_g_words,
                        argument_by_name(func, 'cnw'): context_ng_words,
                        argument_by_name(func, 'qnw'): query_ng_words,
                        argument_by_name(func, 'anw'): answer_ng_words,
                        argument_by_name(func, 'aw' ): answer_words,
                        argument_by_name(func, 'cc' ): context_chars,
                        argument_by_name(func, 'qc' ): query_chars,
                        argument_by_name(func, 'ac' ): answer_chars}
            else:
                yield {} # need to generate empty batch for distributed training

def train(i2w, data_path, model_path, log_file, config_file, restore=False, profiling=False, gen_heartbeat=False):
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
                            rank = C.Communicator.rank(),
                            gen_heartbeat = gen_heartbeat)]

    lr = C.learning_parameter_schedule(training_config['lr'], minibatch_size=None, epoch_size=None)

    ema = {}
#    dummies = []
    for p in z.parameters:
        ema_p = C.constant(0, shape=p.shape, dtype=p.dtype, name='ema_%s' % p.uid)
        ema[p.uid] = ema_p
#        dummies.append(C.reduce_sum(C.assign(ema_p, 0.999 * ema_p + 0.001 * p)))
#    dummy = C.combine(dummies)

    learner = C.adadelta(z.parameters, lr)

    if C.Communicator.num_workers() > 1:
        learner = C.data_parallel_distributed_learner(learner)

    trainer = C.Trainer(z, loss , learner, progress_writers)

    if profiling:
        C.debugging.start_profiler(sync_gpu=True)

    train_data_file = os.path.join(data_path, training_config['train_data'])
    train_data_ext = os.path.splitext(train_data_file)[-1].lower()

    model_file = os.path.join(model_path, model_name)
    model = C.combine(z.outputs + loss.outputs) #this is for validation only
    print(model)
    print(model.outputs[0])
    print(model.outputs[1])
    print(model.outputs[2])
    print(model.outputs[3])

    epoch_stat = {
        'best_val_err' : 100,
        'best_since'   : 0,
        'val_since'    : 0}

    if restore and os.path.isfile(model_file):
        trainer.restore_from_checkpoint(model_file)
        #after restore always re-evaluate
        #TODO replace with rougel with external script(possibly)
        #epoch_stat['best_val_err'] = validate_model(os.path.join(data_path, training_config['val_data']), model, polymath)

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
            trainer.save_checkpoint(model_file)
            epoch_stat['best_since'] += 1
            if epoch_stat['best_since'] > training_config['stop_after']:
                return False
        if profiling:
            C.debugging.enable_profiler()

        return True

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
 #               dummy.eval()
                if num_seq >= epoch_size:
                    break
            if not post_epoch_work(epoch_stat):
                break
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
#                    dummy.eval()
                minibatch_count += 1
            if not post_epoch_work(epoch_stat):
                break

    if profiling:
        C.debugging.stop_profiler()

def symbolic_best_span(begin, end):
    running_max_begin = C.layers.Recurrence(C.element_max, initial_state=-float("inf"))(begin)
    return C.layers.Fold(C.element_max, initial_state=C.constant(-1e+30))(running_max_begin + end)

def validate_model(i2w, test_data, model, polymath):
#    begin_logits = model.outputs[0]
#    end_logits   = model.outputs[1]
#    sparse_to_dense = create_sparse_to_dense(polymath.vocab_size)
    loss         = model.outputs[2]
    print(loss)
    testout = model.outputs[1]  # according to model.shape
    print(testout)
    root = C.as_composite(loss.owner)
    onehot = argument_by_name(root, 'aw')
    mb_source, input_map = create_mb_and_map(root, test_data, polymath, randomize=False, repeat=False)
#    begin_label = argument_by_name(root, 'ab')
#    end_label   = argument_by_name(root, 'ae')
#    onehot = argument_by_name(root, 'aw')

#    begin_prediction = C.sequence.input_variable(1, sequence_axis=begin_label.dynamic_axes[1], needs_gradient=True, name='begin')
#    end_prediction = C.sequence.input_variable(1, sequence_axis=end_label.dynamic_axes[1], needs_gradient=True, name='end')
#    pred_out = C.sequence.input_variable(polymath.vocab_size + 2 , sequence_axis=onehot.dynamic_axes[1], needs_gradient=True, name='predout')
#    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    
#    predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))
#    true_span = C.layers.Recurrence(C.plus)(begin_label - C.sequence.past_value(end_label))
#    common_span = C.element_min(predicted_span, true_span)

 #   predicted_len = C.sequence.reduce_sum(predicted_span)
 #   true_len = C.sequence.reduce_sum(true_span)
 #   common_len = C.sequence.reduce_sum(common_span)
 #   f1 = 2*common_len/(predicted_len+true_len)
 #   exact_match = C.element_min(begin_match, end_match)
 #   precision = common_len/predicted_len
 #   recall = common_len/true_len
 #   overlap = C.greater(common_len, 0)
 #   s = lambda x: C.reduce_sum(x, axis=C.Axis.all_axes())
 #   stats = C.splice(s(f1), s(exact_match), s(precision), s(recall), s(overlap), s(begin_match), s(end_match))
    
    # Evaluation parameters
    minibatch_size = 1024
    num_sequences = 0

    stat = []
    loss_sum = 0
    i=0
    while True:
        data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
        if not data or not (onehot in data) or data[onehot].num_sequences == 0:
            break
        out = model.eval(data, outputs=[onehot,loss], as_numpy=False)
        testloss = out[loss]
 #       outputs = format_sequences(testout, i2w)
        print(testloss)
        print(onehot)
        output = ['a']
        realout  = format_sequences(onehot, i2w)
        print('1')
        if i == 0:
            print(outputs)
            print(realout)
            i=i+1 
             
#        num_total += len(outputs)
#        num_wrong += sum([label != output for output, label in zip(outputs, realout)])
        stat.append(len(outputs) / sum([label != output for output, label in zip(outputs, realout)]))
        print('2')
#        g = best_span_score.grad({begin_prediction:out[begin_logits], end_prediction:out[end_logits]}, wrt=[begin_prediction,end_prediction], as_numpy=False)
#        other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction], begin_label: data[begin_label], end_label: data[end_label]}
#        stat_sum += stats.eval((other_input_map))
        loss_sum += np.sum(testloss.asarray())
        num_sequences += data[onehot].num_sequences
        print('3')
#    stat_avg = stat_sum / num_sequences
    loss_avg = loss_sum / num_sequences
    print(stat)
    print("Validated {} sequences, loss {:.4f}".format(
            num_sequences,
            loss_avg))

    return loss_avg

def create_sparse_to_dense(input_vocab_dim):
    I = C.Constant(np.eye(input_vocab_dim))
    @C.Function
    @C.layers.Signature(InputSequence[C.layers.SparseTensor[input_vocab_dim]])
    def no_op(input):
        return C.times(input, I)
    return no_op

def get_vocab(path):
    # get the vocab for printing output sequences in plaintext
#    vocab = [w.strip() for w in open(path).readlines()]
#    i2w = { i:w for i,w in enumerate(vocab) }
#    w2i = { w:i for i,w in enumerate(vocab) }
    with open(path, 'rb') as vf:
           known, vocab, chars = pickle.load(vf)
    i2w = { i:w for i,w in enumerate(vocab) }
    return i2w

def format_sequences(sequences, i2w):
    print(sequences)
    out = []
    
    for w in sequences:
        print(w)
        x =  np.argmax(w) 
        print(x)
        if x < 127810:
            out.append(i2w[x])
#    for w in sequences:
#        print(w)
    return [" ".join(out)]


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
    polymath = PolyMath(config_file)
    model = C.load_model(os.path.join(model_path, model_file if model_file else model_name))
    loss         = C.as_composite(model.outputs[2].owner)
    output       = model.outputs[1]

    batch_size = 32 # in sequences
    misc = {'rawctx':[], 'ctoken':[], 'answer':[], 'uid':[]}
    tsv_reader = create_tsv_reader(loss, test_data, polymath, batch_size, 1, is_test=True, misc=misc)
    results = {}
    with open('{}_out.json'.format(model_file), 'w', encoding='utf-8') as json_output:
        for data in tsv_reader:
            out = model.eval(data, outputs=[output,loss], as_numpy=False)
            for seq, (raw_text, ctokens, answer, uid) in enumerate(zip(misc['rawctx'], misc['ctoken'], misc['answer'], misc['uid'])):
                predict_answer = output
                results['query_id'] = int(uid)
                results['answers'] = [predict_answer]
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
        model_path = args['outputdir'] + "/models"
    if args['datadir'] is not None:
        data_path = args['datadir']
        
    #C.try_set_default_device(C.gpu(0))

    test_data = args['test']
    test_model = args['model']
    
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
