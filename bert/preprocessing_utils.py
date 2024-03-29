import collections
import itertools
import unicodedata
import numpy as np
import numpy.ma as ma
import multiprocessing as mp
from functools import partial
import time

import mxnet as mx

max_seq_length = 384
max_query_length = 64
doc_stride = 128


SquadBERTFeautre = collections.namedtuple('SquadBERTFeautre', [
    'example_id', 'qas_id', 'doc_tokens', 'valid_length', 'tokens',
    'token_to_orig_map', 'token_is_max_context', 'input_ids', 'p_mask',
    'segment_ids', 'start_position', 'end_position', 'is_impossible'
])


def convert_examples_to_features(example,
                                 tokenizer=None,
                                 cls_token=None,
                                 sep_token=None,
                                 vocab=None,
                                 max_seq_length=384,
                                 doc_stride=128,
                                 max_query_length=64,
                                 cls_index=0):
    """convert the examples to the BERT features"""
    query_tokenized = [cls_token] + tokenizer(
        example.question_text)[:max_query_length]
    # tokenize paragraph and get start/end position of the answer in tokenized paragraph
    tok_start_position, tok_end_position, all_doc_tokens, _, tok_to_orig_index = \
        tokenize_and_align_positions(example.doc_tokens,
                                     example.start_position,
                                     example.end_position,
                                     tokenizer)
    # get doc spans using sliding window
    doc_spans, doc_spans_indices = get_doc_spans(
        all_doc_tokens, max_seq_length - len(query_tokenized) - 2, doc_stride)

    if not example.is_impossible:
        (tok_start_position, tok_end_position) = improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)
        # get the new start/end position
        positions = [
            align_position2doc_spans([tok_start_position, tok_end_position],
                                     doc_idx,
                                     offset=len(query_tokenized) + 1,
                                     default_value=0)
            for doc_idx in doc_spans_indices
        ]
    else:
        # if the question is impossible to answer, set the start/end position to cls index
        positions = [[cls_index, cls_index] for _ in doc_spans_indices]

    # record whether the tokens in a docspan have max context
    token_is_max_context = [{
        len(query_tokenized) + p:
        check_is_max_context(doc_spans_indices, i, p + doc_spans_indices[i][0])
        for p in range(len(doc_span))
    } for (i, doc_span) in enumerate(doc_spans)]

    token_to_orig_map = [{
        len(query_tokenized) + p + 1:
        tok_to_orig_index[p + doc_spans_indices[i][0]]
        for p in range(len(doc_span))
    } for (i, doc_span) in enumerate(doc_spans)]

    # get sequence features: tokens, segment_ids, p_masks
    seq_features = [
        concat_sequences([query_tokenized, doc_span], [[sep_token]] * 2)
        for doc_span in doc_spans
    ]

    features = []
    for (tokens, segment_ids, p_mask), (start, end), is_max, t2o in zip(
            seq_features, positions, token_is_max_context, token_to_orig_map):
        input_ids = vocab[tokens]
        # while len(input_ids) < max_seq_length:
        #    input_ids.append(0)
        #    segment_ids.append(0)
        features.append(SquadBERTFeautre(example_id=example.example_id,
                                         qas_id=example.qas_id,
                                         doc_tokens=example.doc_tokens,
                                         valid_length=len(tokens),
                                         tokens=tokens,
                                         token_to_orig_map=t2o,
                                         token_is_max_context=is_max,
                                         input_ids=input_ids,
                                         p_mask=p_mask,
                                         segment_ids=segment_ids,
                                         start_position=start,
                                         end_position=end,
                                         is_impossible=example.is_impossible))
    return features


def preprocess_dataset(tokenizer,
                       dataset,
                       vocab=None,
                       max_seq_length=384,
                       doc_stride=128,
                       max_query_length=64,
                       input_features=True,
                       num_workers=4,
                       load_from_pickle=False,
                       feature_file=None,
                       for_calibration=False):
    """Loads a dataset into features"""
    vocab = tokenizer.vocab if vocab is None else vocab
    trans = partial(convert_examples_to_features,
                    tokenizer=tokenizer,
                    cls_token=vocab.cls_token,
                    sep_token=vocab.sep_token,
                    vocab=vocab,
                    max_seq_length=max_seq_length,
                    doc_stride=doc_stride,
                    max_query_length=max_query_length)
    pool = mp.Pool(num_workers)
    start = time.time()
    if not load_from_pickle:
        example_trans = partial(convert_squad_examples,
                                is_training=input_features)
        # convert the raw dataset into raw features
        examples = pool.map(example_trans, dataset)
        raw_features = pool.map(trans, examples)
        if feature_file:
            with open(feature_file, 'wb') as file:
                pickle.dump(list(raw_features), file)
    else:
        assert feature_file, 'feature file should be provided.'
        with open(feature_file, 'wb') as file:
            raw_features = pickle.load(file)

    if input_features:
        # convert the full features into the training features
        # Note that we will need the full features to make evaluation
        # Due to using sliding windows in data preprocessing,
        # we will have multiple examples for a single entry after processed.
        # Thus we need to flatten it for training.
        data_feature = mx.gluon.data.SimpleDataset(
            list(itertools.chain.from_iterable(raw_features)))
        if for_calibration:
            data_feature = data_feature.transform(lambda *example: (
                example[7],  # inputs_id
                example[9],  # segment_ids
                example[3],  # valid_length,
                example[10]))  # start_position,
        else:
            data_feature = data_feature.transform(lambda *example: (
                example[0],  # example_id
                example[7],  # inputs_id
                example[9],  # segment_ids
                example[3],  # valid_length,
                example[10],  # start_position,
                example[11]))  # end_position
    else:
        data_feature = mx.gluon.data.SimpleDataset(list(raw_features))

    end = time.time()
    pool.close()
    print('Done! Transform dataset costs %.2f seconds.' % (end - start))
    return data_feature


def concat_sequences(seqs, separators, seq_mask=0, separator_mask=1):
    """Concatenate sequences in a list into a single sequence, using specified separators.

    Example 1:
    seqs: [['is', 'this' ,'jacksonville', '?'], ['no' ,'it' ,'is' ,'not', '.']]
    separator: [[SEP], [SEP], [CLS]]
    seq_mask: 0
    separator_mask: 1
    Returns:
       tokens:      is this jacksonville ? [SEP] no it is not . [SEP] [CLS]
       segment_ids: 0  0    0            0  0    1  1  1  1   1 1     2
       p_mask:      0  0    0            0  1    0  0  0  0   0 1     1

    Example 2:
    separator_mask can also be a list.
    seqs: [['is', 'this' ,'jacksonville', '?'], ['no' ,'it' ,'is' ,'not', '.']]
    separator: [[SEP], [SEP], [CLS]]
    seq_mask: 0
    separator_mask: [[1], [1], [0]]

    Returns:
       tokens:     'is this jacksonville ? [SEP] no it is not . [SEP] [CLS]'
       segment_ids: 0  0    0            0  0    1  1  1  1   1 1     2
       p_mask:      1  1    1            1  1    0  0  0  0   0 1     0

    Example 3:
    seq_mask can also be a list.
    seqs: [['is', 'this' ,'jacksonville', '?'], ['no' ,'it' ,'is' ,'not', '.']]
    separator: [[SEP], [SEP], [CLS]]
    seq_mask: [[1, 1, 1, 1], [0, 0, 0, 0, 0]]
    separator_mask: [[1], [1], [0]]

    Returns:
       tokens:     'is this jacksonville ? [SEP] no it is not . [SEP] [CLS]'
       segment_ids: 0  0    0            0  0    1  1  1  1   1 1     2
       p_mask:      1  1    1            1  1    0  0  0  0   0 1     0

    Parameters
    ----------
    seqs : list
        sequences to be concatenated
    separator : list
        The special tokens to separate sequences.
    seq_mask : int or list
        A single mask value for all sequence items or a list of values for each item in sequences
    separator_mask : int or list
        A single mask value for all separators or a list of values for each separator

    Returns
    -------
    np.array: input token ids in 'int32', shape (batch_size, seq_length)
    np.array: segment ids in 'int32', shape (batch_size, seq_length)
    np.array: mask for special tokens
    """
    assert isinstance(seqs, collections.abc.Iterable) and len(seqs) > 0
    assert isinstance(seq_mask, (list, int))
    assert isinstance(separator_mask, (list, int))
    concat = sum((seq + sep for sep, seq in itertools.zip_longest(separators, seqs, fillvalue=[])),
                 [])
    segment_ids = sum(
        ([i] * (len(seq) + len(sep))
         for i, (sep, seq) in enumerate(itertools.zip_longest(separators, seqs, fillvalue=[]))),
        [])
    if isinstance(seq_mask, int):
        seq_mask = [[seq_mask] * len(seq) for seq in seqs]
    if isinstance(separator_mask, int):
        separator_mask = [[separator_mask] * len(sep) for sep in separators]

    p_mask = sum((s_mask + mask for sep, seq, s_mask, mask in itertools.zip_longest(
        separators, seqs, seq_mask, separator_mask, fillvalue=[])), [])
    return concat, segment_ids, p_mask


def tokenize_and_align_positions(origin_text, start_position, end_position, tokenizer):
    """Tokenize the text and align the origin positions to the corresponding position.

    Parameters
    ----------
    origin_text : list
        list of tokens to be tokenized.
    start_position : int
        Start position in the origin_text
    end_position : int
        End position in the origin_text
    tokenizer : callable function, e.g., BERTTokenizer.

    Returns
    -------
    int: Aligned start position
    int: Aligned end position
    list: tokenized text
    list: map from the origin index to the tokenized sequence index
    list: map from tokenized sequence index to the origin index
    """
    orig_to_tok_index = []
    tok_to_orig_index = []
    tokenized_text = []
    for (i, token) in enumerate(origin_text):
        orig_to_tok_index.append(len(tokenized_text))
        sub_tokens = tokenizer(token)
        tokenized_text += sub_tokens
        tok_to_orig_index += [i] * len(sub_tokens)

    start_position = orig_to_tok_index[start_position]
    end_position = orig_to_tok_index[end_position + 1] - 1 if end_position < len(origin_text) - 1  \
        else len(tokenized_text) - 1
    return start_position, end_position, tokenized_text, orig_to_tok_index, tok_to_orig_index


def get_doc_spans(full_doc, max_length, doc_stride):
    """Obtain document spans by sliding a window across the document

    Parameters
    ----------
    full_doc: list
        The origin doc text
    max_length: max_length
        Maximum size of a doc span
    doc_stride: int
        Step of sliding window

    Returns
    -------
    list: a list of processed doc spans
    list: a list of start/end index of each doc span
    """
    doc_spans = []
    start_offset = 0
    while start_offset < len(full_doc):
        length = min(max_length, len(full_doc) - start_offset)
        end_offset = start_offset + length
        doc_spans.append(
            (full_doc[start_offset:end_offset], (start_offset, end_offset)))
        if start_offset + length == len(full_doc):
            break
        start_offset += min(length, doc_stride)
    return list(zip(*doc_spans))


def align_position2doc_spans(positions, doc_spans_indices, offset=0, default_value=-1,
                             all_in_span=True):
    """Align original positions to the corresponding document span positions

    Parameters
    ----------
    positions: list or int
        A single or a list of positions to be aligned
    dic_spans_indices: list or tuple
        (start_position, end_position)
    offset: int
        Offset of aligned positions. Sometimes the doc spans would be added
        after a question text, in this case, the new position should add
        len(question_text)
    default_value: int
        The default value to return if the positions are not in the doc span.
    all_in_span: bool
        If set to True, then as long as one position is out of span, all positions
        would be set to default_value.

    Returns
    -------
    list: a list of aligned positions
    """
    if not isinstance(positions, list):
        positions = [positions]
    doc_start, doc_end = doc_spans_indices
    if all_in_span and not all([p in range(doc_start, doc_end) for p in positions]):
        return [default_value] * len(positions)
    new_positions = [
        p - doc_start +
        offset if p in range(doc_start, doc_end) else default_value
        for p in positions
    ]
    return new_positions


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer.

    The SQuAD annotations are character based. We first project them to
    whitespace-tokenized words. But then after WordPiece tokenization, we can
    often find a "better match". For example:

      Question: What year was John Smith born?
      Context: The leader was John Smith (1895-1943).
      Answer: 1895

    The original whitespace-tokenized answer will be "(1895-1943).". However
    after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    the exact answer, 1895.

    However, this is not always possible. Consider the following:

      Question: What country is the top exporter of electornics?
      Context: The Japanese electronics industry is the lagest in the world.
      Answer: Japan

    In this case, the annotator chose "Japan" as a character sub-span of
    the word "Japanese". Since our WordPiece tokenizer does not split
    "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    in SQuAD, but does happen.

    Parameters
    ----------
    doc_tokens: list
        A list of doc tokens
    input_start: int
        start position of the answer
    input_end: int
        end position of the answer
    tokenizer: callable function
    orig_answer_text: str
        origin answer text.
    Returns
    -------
    tuple: a tuple of improved start position and end position
    """
    tok_answer_text = ' '.join(tokenizer(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = ' '.join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.

    Because of the sliding window approach taken to scoring documents, a single
    token can appear in multiple documents. E.g.
     Doc: the man went to the store and bought a gallon of milk
     Span A: the man went to the
     Span B: to the store and bought
     Span C: and bought a gallon of
     ...

    Now the word 'bought' will have two scores from spans B and C. We only
    want to consider the score with "maximum context", which we define as
    the *minimum* of its left and right context (the *sum* of left and
    right context will always be the same, of course).

    In the example the maximum context for 'bought' would be span C since
    it has 1 left context and 3 right context, while span B has 4 left context
    and 0 right context.

    Note that position is the absolute position in the origin text.

    Parameters
    ----------
    doc_spans: list
        A list of doc spans
    cur_span_index: int
        The index of doc span to be checked in doc_spans.
    position: int
        Position of the token to be checked.

    Returns
    -------
    bool: True if the token has 'max context'.
    """
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        start, end = doc_span
        end -= 1
        length = end - start + 1
        if position < start:
            continue
        if position > end:
            continue
        num_left_context = position - start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


SquadExample = collections.namedtuple('SquadExample', [
    'qas_id', 'question_text', 'paragraph_text', 'doc_tokens', 'example_id', 'orig_answer_text',
    'start_position', 'end_position', 'start_offset', 'end_offset', 'is_impossible'
])


def convert_squad_examples(record, is_training):
    """read a single entry of gluonnlp.data.SQuAD and convert it to an example.

    Parameters
    ----------
    record: list
        An entry of gluonnlp.data.SQuAD
    is_training: bool
        If the example is used for training,
        then a rough start/end position will be generated

    Returns
    -------
    SquadExample: An instance of SquadExample
    """
    example_id = record[0]
    qas_id = record[1]
    question_text = record[2]
    paragraph_text = record[3]
    orig_answer_text = record[4][0] if record[4] else ''
    answer_offset = record[5][0] if record[5] else ''
    is_impossible = record[6] if len(record) == 7 else False

    answer_length = len(orig_answer_text)
    doc_tokens = []

    char_to_word_offset = []
    prev_is_whitespace = True

    for c in paragraph_text:
        if str.isspace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    if not is_training:
        start_position = -1
        end_position = -1
    else:
        start_position = char_to_word_offset[answer_offset] if not is_impossible else -1
        end_position = char_to_word_offset[answer_offset + answer_length -
                                           1] if not is_impossible else -1
    answer_offset = -1 if is_impossible else answer_offset
    example = SquadExample(
        qas_id=qas_id, question_text=question_text, paragraph_text=paragraph_text,
        doc_tokens=doc_tokens, example_id=example_id, orig_answer_text=orig_answer_text,
        start_position=start_position, end_position=end_position, start_offset=answer_offset,
        end_offset=answer_offset + len(orig_answer_text) - 1, is_impossible=is_impossible)
    return example


def _preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
    """Remove space, convert to lower case, keep accents.

    Parameters
    ----------
    inputs: str
        input string
    lower: bool
        If convert the input string to lower case.
    remove_space: bool
        If remove the spaces in the input string.
    keep_accents: bool
        If keep accents in the input string.
    Returns
    -------
    str: processed input string
    """
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace('``', '"').replace('\'\'', '"')
    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()
    return outputs


def _convert_index(index, pos, M=None, is_start=True):
    """Working best with _lcs_match(), convert the token index to origin text index"""
    if index[pos] is not None:
        return index[pos]
    N = len(index)
    rear = pos
    while rear < N - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if M is not None and index[front] < M - 1:
            if is_start:
                return index[front] + 1
            else:
                return M - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]


def _lcs_match(max_dist, seq1, seq2, max_seq_length=1024, lower=False):
    """Longest common sequence match.

    unlike standard LCS, this is specifically optimized for the setting
    because the mismatch between sentence pieces and original text will be small

    Parameters
    ----------
    max_dist: int
        The max distance between tokens to be considered.
    seq1: list
        The first sequence to be matched.
    seq2: list
        The second sequence to be matched.
    lower: bool
        If match the lower-cased tokens.
    Returns
    -------
    numpyArray: Token-wise lcs matrix f. Shape of ((max(len(seq1), 1024), max(len(seq2), 1024))
    Map: The dp path in matrix f.
        g[(i ,j)] == 2 if token_i in seq1 matches token_j in seq2.
        g[(i, j)] == 1 if token_i in seq1 matches token_{j-1} in seq2.
        g[(i, j)] == 0 of token_{i-1} in seq1 matches token_j in seq2.
    """
    f = np.zeros((max(len(seq1), max_seq_length), max(len(seq2), max_seq_length)),
                 dtype=np.float32)
    g = {}
    for i, token in enumerate(seq1):
        for j in range(i - max_dist, i + max_dist):
            if j >= len(seq2) or j < 0:
                continue

            if i > 0:
                g[(i, j)] = 0
                f[i, j] = f[i - 1, j]

            if j > 0 and f[i, j - 1] > f[i, j]:
                g[(i, j)] = 1
                f[i, j] = f[i, j - 1]

            f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
            if (_preprocess_text(token, lower=lower, remove_space=False) == seq2[j]
                    and f_prev + 1 > f[i, j]):
                g[(i, j)] = 2
                f[i, j] = f_prev + 1
    return f, g
