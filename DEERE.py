import os
os.environ['TF_KERAS']='1'
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import sparse_multilabel_categorical_crossentropy

from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from tqdm import tqdm
from metric import *

maxlen = 2000
batch_size = 1

train_data = load_data(r'Data\train.json')
valid_data = load_data(r'Data\dev.json')
test_data = load_data(r'Data\test.json')
def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=maxlen)
            # create triplets {(s, o, p)}
            spoes = set()
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = relations2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                sh = search(s, token_ids)
                oh = search(o, token_ids)
                if sh != -1 and oh != -1:
                    spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))
            # create labels
            entity_labels = [set() for _ in range(2)]
            head_labels = [set() for _ in range(len(relations2id))]
            tail_labels = [set() for _ in range(len(relations2id))]
            for sh, st, p, oh, ot in spoes:
                entity_labels[0].add((sh, st))
                entity_labels[1].add((oh, ot))
                head_labels[p].add((sh, oh))
                tail_labels[p].add((st, ot))
            for label in entity_labels + head_labels + tail_labels:
                if not label:  #one label at least
                    label.add((0, 0))  #fill with (0,0)
            entity_labels = sequence_padding([list(l) for l in entity_labels])
            head_labels = sequence_padding([list(l) for l in head_labels])
            tail_labels = sequence_padding([list(l) for l in tail_labels])
            # create batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_entity_labels = sequence_padding(
                    batch_entity_labels, seq_dims=2
                )
                batch_head_labels = sequence_padding(
                    batch_head_labels, seq_dims=2
                )
                batch_tail_labels = sequence_padding(
                    batch_tail_labels, seq_dims=2
                )
                yield [batch_token_ids, batch_segment_ids], [
                    batch_entity_labels, batch_head_labels, batch_tail_labels
                ]
                batch_token_ids, batch_segment_ids = [], []
                batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []

#compute crossentropy loss for globalpointer
def globalpointer_crossentropy(y_true, y_pred):
    shape = K.shape(y_pred)
    y_true =K.cast( y_true[..., 0], K.floatx()) * K.cast(shape[2], K.floatx()) +K.cast( y_true[..., 1], K.floatx())
    y_pred = K.reshape(y_pred, (shape[0], -1, K.prod(shape[2:])))
    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, True)
    return K.mean(K.sum(loss, axis=1))

# load pre-trained language model
base = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='roformer',
    return_keras_model=False,
)
#entity head/tail pair recognition
entity_output = GlobalPointer(heads=2, head_size=64)(base.model.output)
#subject/object heads pair recognition
head_output = GlobalPointer(
    heads=len(relations2id), head_size=64, RoPE=False, tril_mask=False
)(base.model.output)
#subject/object tails pair recognition
tail_output = GlobalPointer(
    heads=len(relations2id), head_size=64, RoPE=False, tril_mask=False
)(base.model.output)
outputs = [entity_output, head_output, tail_output]

# create model
AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=1e-5)
model = keras.models.Model(base.model.inputs, outputs)
model.compile(loss=globalpointer_crossentropy, optimizer=optimizer)
model.summary()


def extract_spoes_with_score(text, threshold=0):
    """extract triplet with score from text
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    outputs = model.predict([token_ids, segment_ids])
    outputs = [o[0] for o in outputs]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes_with_score = {}
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)
            for p in ps:
                score = outputs[1][p, sh, oh] + outputs[2][p, st, ot]  #relation strength score
                s = text[mapping[sh][0]:mapping[st][-1] + 1]
                o = text[mapping[oh][0]:mapping[ot][-1] + 1]
                spo = ((s,(sh,st)), id2relation[p],(o,(oh,ot)))
                spoes_with_score[spo] = score
    return spoes_with_score

def get_highest_score_spo(spoes_with_score):
    map = {}
    for spo,score in spoes_with_score.items():
        s,p,o = spo
        if (s,p) not in map or map[(s,p)][1] < score:
            map[(s,p)] = (o,score)
    return [(s,p,o) for (s,p),(o,score) in map.items()]

def evaluate(data,gold_events_list,file):
    """ compute f1, precision and recall
    """
    spoes_with_score_list=[]
    X, Y, Z = 1e-10, 1e-10, 1e-10
    pbar = tqdm()
    pred_events_list = []
    for d in data:
        text =d['text']
        spoes_with_score = extract_spoes_with_score(text)
        R = set([spo for spo in spoes_with_score])
        R = set([(s[0],p,o[0]) for s,p,o in R])     #remove location information
        T = set([spo for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        spoes = {(s[0],p,o[0]):spoes_with_score[(s,p,o)] for s,p,o in spoes_with_score}
        spoes_with_score_list.append(spoes)
        pred_events_list.append(convert_spoes_to_events_by_co_event(spoes))
    pbar.close()
    evaluate_event_all(pred_events_list, gold_events_list )
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_data,gold_dev_events_list,'dev_pred.json')
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('best_model.weights')
        optimizer.reset_old_weights()
        print(
            'dev f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data,gold_test_events_list,'test_pred.json')
        print(
            'test f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )

#training
train_generator = data_generator(train_data, batch_size)
evaluator = Evaluator()
model.fit(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=20,
    callbacks=[evaluator]
)

#evaluate
model.load_weights(r'best_model.weights')
f1, precision, recall = evaluate(test_data,gold_test_events_list,'test_pred.json')
print(
    'f1: %.5f, precision: %.5f, recall: %.5f' %
    (f1, precision, recall)
)
