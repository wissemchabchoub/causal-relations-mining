from inferer_src.src.tasks.infer import infer_from_trained

import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np
import string
import pandas as pd
import itertools
import spacy
import networkx as nx
from itertools import combinations, chain
import pickle
from tqdm import tqdm
import logging
import nltk.data
import json
from spacy.lang.en import English
import regex as re

nlp = spacy.load('en_core_web_sm', disable=['attribute_ruler', 'lemmatizer'])

logging.basicConfig(filename='log.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


class Causality():
    def __init__(self, corpus_df=None, split_spacy=False, word_augmentation=False):

        # Load inferer
        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        args = Namespace(task=1, use_pretrained_blanks=0, num_classes=3, batch_size=64, gradient_acc_steps=2, max_norm=1.0, fp16=0, num_epochs=25, lr=0.00007, model_no=0,
                         model_size='bert-base-uncased', train=0, infer=0)

        self.inferer = infer_from_trained(args=args, detect_entities=False)

        # Load sentences

        if corpus_df is None:
            corpus_df = self.get_sample_corpus_df()
            if split_spacy == True:
                self.sentences = split_sentences_spacy(corpus_df)
            else:
                self.sentences = split_sentences(corpus_df)

        elif type(corpus_df) == str:

            corpus_df = pickle.load(open(corpus_df, "rb"))
            if split_spacy == True:
                self.sentences = split_sentences_spacy(corpus_df)
            else:
                self.sentences = split_sentences(corpus_df)

        else:
            if split_spacy == True:
                self.sentences = split_sentences_spacy(corpus_df)
            else:
                self.sentences = split_sentences(corpus_df)

        logging.info('Articles: {}'.format(len(corpus_df)))
        logging.info('Sentences: {}'.format(len(self.sentences)))

        # read the up_list file back into a Python list object
        with open('data/up_list.txt', 'r') as f:
            UP_LIST = json.loads(f.read())

        # read the down_list file back into a Python list object
        with open('data/down_list.txt', 'r') as f:
            DOWN_LIST = json.loads(f.read())

        if word_augmentation == True:
            UP_LIST = list(set(self.word_augmentation(UP_LIST)))
            DOWN_LIST = list(set(self.word_augmentation(DOWN_LIST)))

        stemmer = SnowballStemmer("english")
        stemmer_ = LancasterStemmer()

        self.up_list = list(
            set([stemmer_.stem(stemmer.stem(word)) for word in UP_LIST]))
        self.down_list = list(
            set([stemmer_.stem(stemmer.stem(word)) for word in DOWN_LIST]))
        try:
            self.down_list.remove('expect')
        except:
            pass

    @staticmethod
    def word_augmentation(lst):
        """
        Adds more word to a list of words

        Parameters
        ----------
        lst : list
            list of words

        Returns
        -------
        list
            list for words plus added synonyms
        """

        synonyms = []

        for word in lst:
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())

        lst += synonyms
        lst = list(set(lst))
        return lst

    @staticmethod
    def mark(sentence, e1, e2):
        """
        Marks words containing entity 1 and entity 2 in a sentence

        Parameters
        ----------
        e1 : str
            entity 1
        e1 : str
            entity 2
        sentence : str
            sentence

        Returns
        -------
        str
            marked sentence
        """

        tokens = word_tokenize(sentence)

        new_tokens = []

        b1 = True
        b2 = True
        for token in tokens:
            if re.search(e1, token, re.IGNORECASE) and b1:
                new_tokens += ['[E1]' + token + '[/E1]']
                b1 = False
            elif re.search(e2, token, re.IGNORECASE) and b2:
                new_tokens += ['[E2]' + token + '[/E2]']
                b2 = False
            else:
                new_tokens += [token]

        """tokens = ['[E1]' + token + '[/E1]' if re.search(
            e1, token, re.IGNORECASE) else token for token in tokens]
        
        tokens = ['[E2]' + token + '[/E2]' if re.search(
            e2, token, re.IGNORECASE) else token for token in tokens]"""

        return ' '.join(new_tokens)

    @staticmethod
    def mark_2(sentence, e1, e2):
        """
        changes words containing entity 1 and entity 2 in a sentence with e1 and e2

        Parameters
        ----------
        e1 : str
            entity 1
        e1 : str
            entity 2
        sentence : str
            sentence

        Returns
        -------
        str
            marked sentence
        """

        tokens = word_tokenize(sentence)

        new_tokens = []

        b1 = True
        b2 = True
        for token in tokens:
            if re.search(e1, token, re.IGNORECASE) and b1:
                new_tokens += [e1]
                b1 = False
            elif re.search(e2, token, re.IGNORECASE) and b2:
                new_tokens += [e2]
                b2 = False
            else:
                new_tokens += [token]

        """tokens = ['[E1]' + token + '[/E1]' if re.search(
            e1, token, re.IGNORECASE) else token for token in tokens]
        
        tokens = ['[E2]' + token + '[/E2]' if re.search(
            e2, token, re.IGNORECASE) else token for token in tokens]"""

        return ' '.join(new_tokens)

    @staticmethod
    def unmark(sentence):
        """
        unmarks words between [E1] and [/E1] and [E2] and [/E2] in a sentence and replaces them with e1 and e2

        Parameters
        ----------
        sentence : str
            sentence

        Returns
        -------
        str
            unmarked sentence
        """

        sentence = re.sub(
            r"\[E1\](.*?)\[/E1\]",
            r'\1',
            sentence
        )

        sentence = re.sub(
            r"\[E2\](.*?)\[/E2\]",
            r'\1',
            sentence
        )

        return sentence

    def get_pair_sentences(self, e1, e2, sentences_df):
        """
        Returns list of sentences containing e1 and e2 (in that order) and marks them

        Parameters
        ----------
        e1 : str
            entity 1
        e1 : str
            entity 2
        sentences_df : pd.DataFrame
            df of sentences

        Returns
        -------
        pd.DataFrame
            df with marked sentences
        """

        """filtration_mask = [True if (re.search(rf"\b{e1}", val, re.IGNORECASE) and re.search(rf"\b{e2}", val, re.IGNORECASE) and
                                    re.search(rf"\b{e1}", val, re.IGNORECASE).span()[0] < re.search(rf"\b{e2}", val, re.IGNORECASE).span()[1]) else False for val in sentences_df.sentence]"""

        filtration_mask = []

        for sentence in sentences_df.sentence:
            try:
                if self.contains(e1, e2, sentence):
                    filtration_mask += [True]
                else:
                    filtration_mask += [False]
            except:
                filtration_mask += [False]

        filtered_df = sentences_df.iloc[filtration_mask].reset_index(drop=True)
        filtered_df.sentence = filtered_df.sentence.apply(
            lambda sentence: self.mark(sentence, e1, e2))

        return filtered_df.dropna().reset_index(drop=True)

    def contains(self, e1, e2, sentence):
        boolean = True
        # contains e1, e2 and pos(e2)- pos(e1) > len(e1) + 5
        boolean = boolean and (re.search(rf"\b{e1}", sentence, re.IGNORECASE) and re.search(rf"\b{e2}", sentence, re.IGNORECASE) and
                               re.search(e2, sentence, re.IGNORECASE).span()[0] - re.search(e1, sentence, re.IGNORECASE).span()[0] > len(e1) + 5)

        if boolean:
            sentence_doc = nlp(sentence)
            tags = [token.tag_ for token in sentence_doc]
            tokens = word_tokenize(sentence)

            r1 = re.compile(rf"\b{e1}")
            r2 = re.compile(rf"\b{e2}")

            e1_token_id = tokens.index(list(filter(r1.match, tokens))[0])
            e2_token_id = tokens.index(list(filter(r2.match, tokens))[0])

            # e1 is NN in the sentence and e1 is NN in the sentence
            boolean = boolean and 'NN' in tags[e1_token_id] and 'NN' in tags[e2_token_id]

            return boolean

        return boolean

    def get_relations(self, pair, sentences_df):
        """
        Returns sentences having a causal relation between a pair of entities

        Parameters
        ----------
        pair : list
            pair of entities
        sentences_df : pd.DataFrame
            df of sentences

        Returns
        -------
        relations_df
            df of causal sentences
        """

        def f(sentence):
            try:
                return self.inferer.infer_sentence(sentence, detect_entities=False)
            except:
                return 'long'

        direction_1_sentences = []
        direction_2_sentences = []

        filtered_1 = self.get_pair_sentences(pair[0], pair[1], sentences_df)

        filtered_1['relation'] = filtered_1.sentence.apply(
            lambda sentence: f(sentence))

        filtered_1_direction_1_sentences = filtered_1[filtered_1.relation ==
                                                      'Cause-Effect(e1,e2)']
        filtered_1_direction_2_sentences = filtered_1[filtered_1.relation ==
                                                      'Cause-Effect(e2,e1)']

        filtered_2 = self.get_pair_sentences(pair[1], pair[0], sentences_df)

        filtered_2['relation'] = filtered_2.sentence.apply(
            lambda sentence: f(sentence))

        filtered_2_direction_2_sentences = filtered_2[filtered_2.relation ==
                                                      'Cause-Effect(e1,e2)']
        filtered_2_direction_1_sentences = filtered_2[filtered_2.relation ==
                                                      'Cause-Effect(e2,e1)']

        direction_1_sentences = pd.concat(
            [filtered_1_direction_1_sentences, filtered_2_direction_1_sentences])
        direction_2_sentences = pd.concat(
            [filtered_1_direction_2_sentences, filtered_2_direction_2_sentences])

        print('sentences conataining : ', pair)
        print(len(filtered_1)+len(filtered_2))

        print('number of relations found :')
        print(len(direction_1_sentences), len(direction_2_sentences))

        direction_1_sentences['cause'] = pair[0]
        direction_1_sentences['effect'] = pair[1]
        direction_1_sentences = direction_1_sentences.drop(
            ['relation'], axis=1).reset_index(drop=True)

        direction_2_sentences['cause'] = pair[1]
        direction_2_sentences['effect'] = pair[0]
        direction_2_sentences = direction_2_sentences.drop(
            ['relation'], axis=1).reset_index(drop=True)

        relations_df = pd.concat(
            [direction_1_sentences, direction_2_sentences]).reset_index(drop=True)

        return relations_df.dropna().reset_index(drop=True)

    @staticmethod
    def get_shortest_path(graph, e, lst):
        """
        Returns the length of the shortest path between an entity and a list of words in a network

        Parameters
        ----------
        graph : nx.DAG
            network
        e : str
            entity
        lst : list
            list of words

        Returns
        -------
        int
            length of the shortest path
        """

        paths = []

        for e2 in lst:
            try:
                paths.append(nx.shortest_path_length(
                    graph, source=e, target=e2))
            except:
                pass

        if not paths:
            return 1000

        return np.min(paths)

    def get_direction_updated(self, e1, e2, sentence):
        """
        Deduces the direction of change of e1 and e2 in a sentence

        Parameters
        ----------
        e1 : str
            entity 1
        e1 : str
            entity 2
        sentence : str
            sentence (marked) containing e1 and e2

        Returns
        -------
        bool
            direction of e1 (0 : down , 1 : up)
        bool
            direction of e2 (0 : down , 1 : up)
        """

        sentence = self.unmark(sentence)
        sentence = self.mark_2(sentence, e1, e2)

        # Preprocess

        tokens = word_tokenize(sentence)

        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        words = [word for word in stripped if word.isalpha()]

        sentence = ' '.join(words)

        # Create graph using idx
        doc = nlp(sentence)

        id2txt = {str(token.idx): token.text for token in doc}

        edges = []

        for token in doc:
            for child in token.children:
                edges.append(('{0}'.format(token.idx),
                              '{0}'.format(child.idx)))

        graph = nx.Graph(edges)

        # transform words to idx
        stemmer = SnowballStemmer("english")
        stemmer_ = LancasterStemmer()

        id2txt_stemmed = {key: (w if w in [e1, e2] else stemmer_.stem(
            stemmer.stem(w))) for key, w in id2txt.items()}

        e1_idx = list(id2txt_stemmed.keys())[
            list(id2txt_stemmed.values()).index(e1)]
        e2_idx = list(id2txt_stemmed.keys())[
            list(id2txt_stemmed.values()).index(e2)]

        def list2idx(word):
            try:
                il = np.where(
                    np.array(list(id2txt_stemmed.values())) == word)[0]
                return [list(id2txt_stemmed.keys())[i] for i in il]
            except:
                pass

        up_idx = list(map(list2idx, self.up_list))
        up_idx = list(itertools.chain(*[i for i in up_idx if i]))

        down_idx = list(map(list2idx, self.down_list))
        down_idx = list(itertools.chain(*[i for i in down_idx if i]))

        # find distances
        e1_up = self.get_shortest_path(graph, e1_idx, up_idx)
        e1_down = self.get_shortest_path(graph, e1_idx, down_idx)

        e2_up = self.get_shortest_path(graph, e2_idx, up_idx)
        e2_down = self.get_shortest_path(graph, e2_idx, down_idx)

        if e1_up == 1000 and e1_down == 1000:
            raise Exception("Couldn't find direction")

        elif e2_up == 1000 and e2_down == 1000:
            raise Exception("Couldn't find direction")

        return int(e1_up <= e1_down), int(e2_up <= e2_down)

    def apply_get_direction_updated(self, row):
        try:
            directions = self.get_direction_updated(
                row['cause'], row['effect'], row['sentence'])
        except:
            directions = None
        row['directions'] = directions
        return row

    def get_counts(self, pair, sentences_df):
        relations_df = self.get_relations(pair, sentences_df)
        relations_df['directions'] = None
        relations_df = relations_df.apply(
            self.apply_get_direction_updated, axis=1)

        return relations_df.drop_duplicates(subset=['sentence'], keep='last').dropna().reset_index(drop=True)
        # return relations_df.reset_index(drop=True)

    def create_data(self, nodes1, name='data.p', sentences_df=None):
        """
        Generates dataframe of causalities and JPTs

        Parameters
        ----------
        nodes : list
            list of entities
        sentences_df : df.DataFrame
            list of sentences (default is None)
        name : str
            saving path for the df 

        Returns
        -------
        pd.DataFrame
            dataframe of causalities and JPTs

        """
        # verify if it is a nested list
        for sublist in nodes1:
            if (type(sublist) == list):
                nodes = list(chain.from_iterable(nodes1))
            else:
                nodes = nodes1

        if sentences_df is None:
            sentences_df = self.sentences

        data = pd.DataFrame(
            columns=['date', 'sentence', 'cause', 'effect', 'directions'])

        for e1, e2 in combinations(nodes, r=2):

            for sublist in nodes1:

                if ((e1 in sublist) and (e2 not in sublist)):
                    df_pair = self.get_counts([e1, e2], sentences_df)
                    data = pd.concat([data, df_pair]).reset_index(drop=True)
                else:
                    pass

        pickle.dump(data, open(name, "wb"))

        return data

    @staticmethod
    def get_sample_corpus_df():
        """
        Gets list of articles from nltk reuters corpus_df

        Parameters
        ----------

        Returns
        -------
        pd.DataFrame
            df of articles

        """

        doc_list = np.array(reuters.fileids())

        test_doc = doc_list[['test' in x for x in doc_list]]

        train_doc = doc_list[['training' in x for x in doc_list]]

        test_corpus = [" ".join([t for t in reuters.words(test_doc[t])])
                       for t in range(len(test_doc))]

        train_corpus = [" ".join([t for t in reuters.words(train_doc[t])])
                        for t in range(len(train_doc))]

        corpus = test_corpus + train_corpus

        def preprocess(text):
            text = re.sub(r'\s([?.!"])', r'\1', text).replace(
                '>', '').replace('-', '')
            text = re.sub(' +', ' ', text)
            text = re.sub("(\d(\s|)\.\s\d)", lambda m: m.group(
                1).replace(" ", "").replace('.', ','), text)

            return text

        corpus = list(map(preprocess, corpus))

        corpus_df = pd.DataFrame(columns=['date', 'text'])
        corpus_df.text = corpus
        corpus_df.date = pd.to_datetime('01-01-2021')

        return corpus_df.drop_duplicates(subset=['text'], keep='last').reset_index(drop=True)


def split_sentences_spacy(corpus_df):
    """
    Gets df of sentences from df of articles

    Parameters
    ----------
    corpus_df : pd.DataFrame
        df of articles

    Returns
    -------
    pd.DataFrame
        df of sentences

    """
    def to_text(x): return x.text

    sentences_df = corpus_df.copy()
    #sentences_df = sentences_df.drop_duplicates(subset=['text'], keep='last').dropna().reset_index(drop=True)

    # Stripping everything but alphanumeric chars
    pattern = re.compile("[^a-zA-Z0-9_ ',.:!;()/]", re.UNICODE)
    # remove text within parentheses
    sentences_df.text = sentences_df.text.map(lambda x: re.sub(
        r"\([^()]*\)", " ", re.sub(pattern, '', x)), na_action=None)
    tqdm.pandas()
    sentences_df.text = sentences_df.text.progress_apply(
        lambda text: list(nlp(text).sents))
    # print("nlp.pipe_names",nlp.pipe_names)
    sentences_df = sentences_df.explode('text')
    sentences_df.text = sentences_df.text.apply(to_text)
    sentences_df.text = sentences_df.text.map(lambda x: x.split('\n'))
    sentences_df = sentences_df.explode('text')
    # replace rows with only numbers with nan
    sentences_df.text = sentences_df.text.map(
        lambda x: np.nan if (re.sub('\.$', '', x).isdigit()) else x)
    sentences_df = sentences_df.rename(columns={"text": "sentence"}).dropna()
    sentences_df = sentences_df.loc[sentences_df.sentence.apply(
        lambda sentence: len(sentence.split())) <= 50]
    sentences_df = sentences_df.loc[sentences_df.sentence.apply(
        lambda sentence: len(sentence.split())) > 3]
    sentences_df = sentences_df.drop_duplicates(
        subset=['sentence'], keep='last').dropna().reset_index(drop=True)
    return sentences_df


def split_sentences(corpus_df):
    """
    Gets df of sentences from df of articles

    Parameters
    ----------
    corpus_df : pd.DataFrame
        df of articles

    Returns
    -------
    pd.DataFrame
        df of sentences

    """

    sentences_df = corpus_df.copy()
    #sentences_df = sentences_df.drop_duplicates(subset=['text'], keep='last').dropna().reset_index(drop=True)

    def preprocess(text):
        text = text.replace('*', '.').replace('-', '.').replace(';', '.')

        return text
    sentences_df.text = sentences_df.text.apply(preprocess)

    # Stripping everything but alphanumeric chars
    pattern = re.compile("[^a-zA-Z0-9_ ',.:!;()/]", re.UNICODE)
    # remove text within parentheses
    sentences_df.text = sentences_df.text.map(lambda x: re.sub(
        r"\([^()]*\)", " ", re.sub(pattern, '', x)), na_action=None)

    tqdm.pandas()
    sentences_df.text = sentences_df.text.progress_apply(nltk.sent_tokenize)
    sentences_df = sentences_df.explode('text')

    sentences_df.text = sentences_df.text.map(lambda x: x.split('\n'))
    sentences_df = sentences_df.explode('text')
    # replace rows with only numbers with nan
    sentences_df.text = sentences_df.text.map(
        lambda x: np.nan if (re.sub('\.$', '', x).isdigit()) else x)
    sentences_df = sentences_df.rename(columns={"text": "sentence"}).dropna()
    sentences_df = sentences_df.loc[sentences_df.sentence.apply(
        lambda sentence: len(sentence.split())) <= 50]
    sentences_df = sentences_df.loc[sentences_df.sentence.apply(
        lambda sentence: len(sentence.split())) > 3]
    sentences_df = sentences_df.drop_duplicates(
        subset=['sentence'], keep='last').dropna().reset_index(drop=True)
    return sentences_df
