import nltk
import numpy as np
import pandas as pd
import io
sent_tokens = []

"""
@tagset     : list of tags.
@tag_count  : dictionary of tag count in corpus. 
"""


def get_penn_tree_bank_data():
    from nltk.corpus import treebank

    nltk.download('treebank')
    penn_tree_bank = []

    raw_string = ""

    for ids in treebank.fileids():
        for word, tag in treebank.tagged_words(ids):
            raw_string = raw_string + word + '/' + tag + " "

    # open text file
    text_file = open("data1.txt", "w")

    # write string to file
    text_file.write(raw_string)

    # close file
    text_file.close()


def get_tag_count(sent_tokens=None):

    if sent_tokens is None:
        print("sent_tokens is None please provide the sent_tokens !")
        return None

    tagset = []
    for idx, sent in enumerate(sent_tokens):
        for idx1, token in enumerate(sent):
            l = token.split('/')
            tk, tag = l[0], l[1]
            if tag not in tagset and len(l) == 2:
                tagset.append(tag)

    tag_count = {key: 0 for key in tagset}

    for idx, sent in enumerate(sent_tokens):
        for idx1, token in enumerate(sent):
            l = token.split('/')
            if len(l) == 2:
                tag_count[l[1]] += 1

    return tagset, tag_count


def get_initial_prob(sent_tokens, tagset):
    initial_prob = []
    initial_dict_count = {key: 0 for key in tagset}

    total = len(sent_tokens)

    for sent_token in sent_tokens:
        if len(sent_token) > 1:
            l = sent_token[0].split('/')
            if len(l) == 2:
                tk, t = l[0], l[1]
                initial_dict_count[t] += 1

    # initial_dict_count = {k: v for k, v in sorted(
    #     initial_dict_count.items(), key=lambda item: item[1])}

    # print("pause")
    for idx, tag in enumerate(tagset):
        initial_prob.append(initial_dict_count[tag]/total)

    return initial_prob


def transition_prob(sent_tokens, tagset, tag_count):

    count_matrix = np.zeros(shape=(len(tagset), len(tagset)), dtype=float)
    transition_matrix = np.zeros(shape=(len(tagset), len(tagset)), dtype=float)

    for idx1, sent_token in enumerate(sent_tokens):
        for idx2, token in enumerate(sent_token):
            if idx2 != 0:
                l1 = sent_token[idx2-1].split('/')
                l2 = sent_token[idx2].split('/')
                if len(l1) == 2 and len(l2) == 2:
                    count_matrix[tagset.index(l2[1]), tagset.index(l1[1])] += 1

    for id1, tag1 in enumerate(tagset):
        for id2, tag2 in enumerate(tagset):
            transition_matrix[id1, id2] = count_matrix[id1,
                                                       id2] / tag_count[tag2]

    return transition_matrix


def find_em_prob(obs, tagset, tag_count):
    global sent_tokens

    N = len(obs)
    M = len(tagset)
    prob = np.zeros(shape=(N, M), dtype=float)
    count = np.zeros(shape=(N, M), dtype=float)
    for obs_id, ob in enumerate(obs):
        for idx1, sent_token in enumerate(sent_tokens):
            for idx2, token in enumerate(sent_token):
                l = token.split('/')
                if len(l) == 2 and ob.lower() == l[0].lower():
                    state_id = tagset.index(l[1])
                    count[obs_id][state_id] += 1

    for n in range(len(obs)):
        for m in range(len(tagset)):
            prob[n][m] = (count[n][m]) / (tag_count[tagset[m]])

    return prob


def viterbi_algo(text, tagset, tag_count, initial_prob, trans_prob):
    global sent_tokens
    obs = text.split()
    sequence = []
    em_prob = find_em_prob(obs, tagset, tag_count)
    N = len(obs)
    K = len(tagset)

    vb = np.zeros(shape=(N, K), dtype=float)
    bp = np.zeros(shape=(N, K), dtype=int)

    inital_ob = 0
    for s in range(K):
        vb[inital_ob][s] = initial_prob[s] * em_prob[inital_ob][s]
        bp[inital_ob][s] = -1
    for ob in range(1, N):
        vb[ob-1] = vb[ob-1] * 100
        for s1 in range(K):
            temp = vb[ob-1, :] * trans_prob[s1, :] * em_prob[ob][s1]
            vb[ob][s1] = np.max(temp)
            bp[ob][s1] = np.argmax(temp)

    prev = np.argmax(vb[N-1])
    sequence.append(tagset[prev])
    for ob in range(N-1, 0, -1):
        curr = bp[ob][prev]
        sequence.append(tagset[curr])
        prev = curr

    # sequence = []
    # index = -1
    # prob_list = np.zeros(shape=(1, K), dtype=float)

    # for idx1 in range(0, len(obs)):
    #     prob_list = trans_prob[:, index] * em_prob[idx1, :]
    #     index = np.argmax(prob_list)
    #     sequence.append(tagset[index])

    return sequence


def accuracy(true_label, pred_label):
    t_tokens = true_label.split()
    p_tokens = pred_label.split()
    correct_pred = 0
    wrong_pred = 0
    for i in range(len(t_tokens)):
        l1 = t_tokens[i].split('/')
        l2 = p_tokens[i].split('/')
        if len(l1) == 2 and len(l2) == 2 and l1[1] == l2[1]:
            correct_pred += 1
        else:
            wrong_pred += 1

    acc = correct_pred / len(t_tokens)
    acc = acc * 100
    acc = "{:.2f}".format(acc)

    return str(acc) + ' %'


def main():

    global sent_tokens

    file = io.open('data/train.txt', 'r')
    data = file.read()

    sentences = data.split("./.")

    for sent in sentences:
        sent = sent + ' ./.'
        tokens = sent.split()
        sent_tokens.append(tokens)

    tagset, tag_count = get_tag_count(sent_tokens)

    initial_prob = get_initial_prob(sent_tokens, tagset)

    transition_probability = transition_prob(sent_tokens, tagset, tag_count)

    file1 = io.open('data/test_X.txt', 'r')
    test = file1.read()
    sentences_test = test.split('\n')
    raw_string_pred = ''

    for text in sentences_test:
        tokens = text.split()

        sequence = viterbi_algo(text, tagset, tag_count,
                                initial_prob, transition_probability)

        sequence.reverse()
        for i in range(len(tokens)):
            raw_string_pred = raw_string_pred + \
                tokens[i] + '/' + sequence[i] + ' '

        raw_string_pred += '\n'

    file4 = io.open('data/test_y.txt', 'r')
    test_true = file4.read()

    acc = accuracy(test_true, raw_string_pred)
    print('Accuracy : ', acc)
    file2 = io.open('data/test_pred_v.txt', 'w')
    file2.write(raw_string_pred)
    file2.close()


if __name__ == '__main__':
    main()
