# Preprocessing
import sys
import random
import time
import json
import os
import argparse
from utils import *
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Modeling
from models import BiLSTM_LMCL, large_margin_cosine_loss
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from keras import backend as K

# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    # arguments need to specify
    parser.add_argument("--dataset", type=str, choices=["ATIS", "SNIPS", "CLINC"], required=True,
                        help="The dataset to use, ATIS or SNIPS.")
    parser.add_argument("--proportion", type=int, required=True,
                        help="The proportion of seen classes, range from 0 to 100.")
    parser.add_argument("--seen_classes", type=str, nargs="+", default=None,
                        help="The specific seen classes.")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both",
                        help="Specify running mode: only train, only test or both.")
    parser.add_argument("--setting", type=str, nargs="+", default=None,
                        help="The settings to detect ood samples, e.g. 'lof' or 'gda_lsqr")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="The directory contains model file (.h5), requried when test only.")
    parser.add_argument("--seen_classes_seed", type=int, default=None,
                        help="The random seed to randomly choose seen classes.")
    # default arguments
    parser.add_argument("--gpu_device", type=str, default="1",
                        help="The gpu device to use.")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="The directory to store training models & logs.")
    # model hyperparameters
    parser.add_argument("--embedding_file", type=str,
                        default="glove.6B.300d.txt",
                        help="The embedding file to use.")
    parser.add_argument("--embedding_dim", type=int, default=300,
                        help="The dimension of word embeddings.")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="The max sequence length. When set to None, it will be implied from data.")
    parser.add_argument("--max_num_words", type=int, default=10000,
                        help="The max number of words.")
    # training hyperparameters
    parser.add_argument("--max_epoches", type=int, default=200,
                        help="Max epoches when training.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience when applying early stop.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Mini-batch size for train and validation")
    args = parser.parse_args()
    return args
args = parse_args()

dataset = args.dataset
proportion = args.proportion

EMBEDDING_FILE = args.embedding_file
MAX_SEQ_LEN = args.max_seq_len
MAX_NUM_WORDS = args.max_num_words
EMBEDDING_DIM = args.embedding_dim

df, partition_to_n_row = load_data(dataset)

df['content_words'] = df['text'].apply(lambda s: word_tokenize(s))
texts = df['content_words'].apply(lambda l: " ".join(l)) 

# Do not filter out "," and "."
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~') 

tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
sequences_pad = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

# Train-valid-test split
idx_train = (None, partition_to_n_row['train'])
idx_valid = (partition_to_n_row['train'], partition_to_n_row['train'] + partition_to_n_row['valid'])
idx_test = (partition_to_n_row['train'] + partition_to_n_row['valid'], None)

X_train = sequences_pad[idx_train[0]:idx_train[1]]
X_valid = sequences_pad[idx_valid[0]:idx_valid[1]]
X_test = sequences_pad[idx_test[0]:idx_test[1]]

df_train = df[idx_train[0]:idx_train[1]]
df_valid = df[idx_valid[0]:idx_valid[1]]
df_test = df[idx_test[0]:idx_test[1]]

y_train = df_train.label.reset_index(drop=True)
y_valid = df_valid.label.reset_index(drop=True)
y_test = df_test.label.reset_index(drop=True)
print("train : valid : test = %d : %d : %d" % (X_train.shape[0], X_valid.shape[0], X_test.shape[0]))


n_class = y_train.unique().shape[0]
n_class_seen = round(n_class * proportion/100)

if args.seen_classes is None:
    if args.seen_classes_seed is not None:
        random.seed(args.seen_classes_seed)
        y_cols = y_train.unique()
        y_cols_lst = list(y_cols)
        random.shuffle(y_cols_lst)
        y_cols_seen = y_cols_lst[:n_class_seen]
        y_cols_unseen = y_cols_lst[n_class_seen:]
    else:
        # Original implementation
        weighted_random_sampling = False
        if weighted_random_sampling:
            y_cols = y_train.unique()
            y_vc = y_train.value_counts()
            y_vc = y_vc / y_vc.sum()
            y_cols_seen = np.random.choice(y_vc.index, n_class_seen, p=y_vc.values, replace=False)
            y_cols_unseen = [y_col for y_col in y_cols if y_col not in y_cols_seen]
        else:
            y_cols = list(y_train.unique())
            y_cols_seen = random.sample(y_cols, n_class_seen)
            y_cols_unseen = [y_col for y_col in y_cols if y_col not in y_cols_seen]
else:
    y_cols = y_train.unique()
    y_cols_seen = [y_col for y_col in y_cols if y_col in args.seen_classes]
    y_cols_unseen = [y_col for y_col in y_cols if y_col not in args.seen_classes]
print(y_cols_seen)

train_seen_idx = y_train[y_train.isin(y_cols_seen)].index
valid_seen_idx = y_valid[y_valid.isin(y_cols_seen)].index

X_train_seen = X_train[train_seen_idx]
y_train_seen = y_train[train_seen_idx]
X_valid_seen = X_valid[valid_seen_idx]
y_valid_seen = y_valid[valid_seen_idx]

le = LabelEncoder()
le.fit(y_train_seen)
y_train_idx = le.transform(y_train_seen)
y_valid_idx = le.transform(y_valid_seen)
y_train_onehot = to_categorical(y_train_idx)
y_valid_onehot = to_categorical(y_valid_idx)

y_test_mask = y_test.copy()
y_test_mask[y_test_mask.isin(y_cols_unseen)] = 'unseen'

train_data = (X_train_seen, y_train_onehot)
valid_data = (X_valid_seen, y_valid_onehot)
test_data = (X_test, y_test_mask)

if args.mode in ["train", "both"]:
    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    set_allow_growth(device=args.gpu_device)

    timestamp = str(time.time()) # strftime("%m%d%H%M")
    output_dir = os.path.join(args.output_dir, f"{dataset}-{proportion}-{timestamp}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, "seen_classes.txt"), "w") as f_out:
        f_out.write("\n".join(le.classes_))
    with open(os.path.join(output_dir, "unseen_classes.txt"), "w") as f_out:
        f_out.write("\n".join(y_cols_unseen))

    print("Load pre-trained GloVe embedding...")
    MAX_FEATURES = min(MAX_NUM_WORDS, len(word_index)) + 1  # +1 for PAD

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    filepath = os.path.join(output_dir, 'model.h5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, 
                                save_best_only=True, mode='auto', save_weights_only=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=args.patience, mode='auto') 
    callbacks_list = [checkpoint, early_stop]

    model = BiLSTM_LMCL(MAX_SEQ_LEN, MAX_FEATURES, EMBEDDING_DIM, n_class_seen, None, embedding_matrix)
    model.fit(train_data[0], train_data[1], epochs=args.max_epoches, batch_size=args.batch_size, 
                        validation_data=valid_data, shuffle=True, verbose=1, callbacks=callbacks_list)

if args.mode in ["test", "both"]:

    if args.mode == "test":
        model_dir = args.model_dir
        model = load_model(os.path.join(model_dir, "model.h5"),
            custom_objects={"large_margin_cosine_loss": large_margin_cosine_loss})
    else:
        model_dir = output_dir

    y_pred_proba = model.predict(test_data[0])
    y_pred_proba_train = model.predict(train_data[0])
    classes = list(le.classes_) + ['unseen']
    get_deep_feature = Model(inputs=model.input, 
                            outputs=model.layers[-3].output)
    feature_test = get_deep_feature.predict(test_data[0])
    feature_train = get_deep_feature.predict(train_data[0])

    for setting in args.setting:
        pred_dir = os.path.join(model_dir, f"{setting}")
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        setting_fields = setting.split("_")
        ood_method = setting_fields[0]

        assert ood_method in ("lof", "gda", "msp")

        if ood_method == "lof":
            method = 'LOF (LMCL)'
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
            lof.fit(feature_train)

            y_pred_lof = pd.Series(lof.predict(feature_test))
            test_info = get_test_info(texts=texts[idx_test[0]:idx_test[1]],
                                    label=y_test,
                                    label_mask=y_test_mask,
                                    softmax_prob=y_pred_proba,
                                    softmax_classes=list(le.classes_),
                                    lof_result=y_pred_lof,
                                    save_to_file=True,
                                    output_dir=pred_dir)
            df_seen = pd.DataFrame(y_pred_proba, columns=le.classes_)
            df_seen['unseen'] = 0

            y_pred = df_seen.idxmax(axis=1)
            y_pred[y_pred_lof[y_pred_lof==-1].index]='unseen'
            cm = confusion_matrix(test_data[1], y_pred, classes)
            f, f_seen, f_unseen = get_score(cm)
        elif ood_method == "gda":
            solver = setting_fields[1] if len(setting_fields) > 1 else "lsqr"
            threshold = setting_fields[2]
            distance_type = setting_fields[3] if len(setting_fields) > 3 else "mahalanobis"
            assert solver in ("svd", "lsqr")
            assert distance_type in ("mahalanobis", "euclidean")

            method = 'GDA (LMCL)'
            gda = LinearDiscriminantAnalysis(solver=solver, shrinkage=None, store_covariance=True)
            gda.fit(feature_train, y_train_seen)

            threshold = float(threshold)

            y_pred = pd.Series(gda.predict(feature_test))
            gda_result = confidence(feature_test, gda.means_, distance_type, gda.covariance_)
            test_info = get_test_info(texts=texts[idx_test[0]:idx_test[1]],
                                    label=y_test,
                                    label_mask=y_test_mask,
                                    softmax_prob=y_pred_proba,
                                    softmax_classes=list(le.classes_),
                                    gda_result=gda_result,
                                    gda_classes=gda.classes_,
                                    save_to_file=True,
                                    output_dir=pred_dir)
            y_pred_score = pd.Series(gda_result.min(axis=1))
            y_pred[y_pred_score[y_pred_score > threshold].index] = 'unseen'
            cm = confusion_matrix(test_data[1], y_pred, classes)
            f, f_seen, f_unseen = get_score(cm)
        elif ood_method == "msp":
            threshold = setting_fields[1]
            method = 'MSP (LMCL)'
            
            threshold = float(threshold)
            
            df_seen = pd.DataFrame(y_pred_proba, columns=le.classes_)
            df_seen['unseen'] = 0

            y_pred = df_seen.idxmax(axis=1)
            y_pred_score = df_seen.max(axis=1)
            y_pred[y_pred_score[y_pred_score < threshold].index]='unseen'
            cm = confusion_matrix(test_data[1], y_pred, classes)
            f, f_seen, f_unseen = get_score(cm)
