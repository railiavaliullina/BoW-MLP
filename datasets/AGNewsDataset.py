import torch
import time
import numpy as np
import spacy
from collections import Counter
import re
import time

from utils.dataframes_handler import read_file, save_file


class AGNewsDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_type):
        """
        Class for reading, preprocessing, encoding and sampling data.
        :param cfg: dataset config
        :param dataset_type: 'train' or 'test' data type
        """
        self.cfg = cfg
        self.dataset_type = dataset_type

        # read data
        self.read_data()

        # load preprocessed data if possible, else preprocess
        if self.cfg.load_preprocessed_data:
            self.load_preprocessed_data()
        else:
            self.preprocess_data()

    def read_data(self):
        """
        Reads source data.
        """
        csv = read_file(self.cfg.dataset_path + f'{self.dataset_type}.csv')
        self.labels = csv['Class Index'].to_numpy() - 1
        self.texts = csv['Description'].to_list()
        self.texts_num = self.__len__()

    def load_preprocessed_data(self):
        """
        Loads preprocessed and saved data.
        """
        self.preprocessed_texts = read_file(self.cfg.preprocessed_dataset_path +
                                            f'{self.dataset_type}_preprocessed.pickle').preprocessed_text.to_list()
        self.vocab = read_file(self.cfg.preprocessed_dataset_path + f'vocab.pickle')
        self.vocab = {k: v for k, v in zip(self.vocab['token'], self.vocab['frequency'])}
        self.vocab_size = len(self.vocab)
        self.vocab_words = np.asarray(list(self.vocab.keys()))
        self.weighted_vector_values_df = read_file(self.cfg.preprocessed_dataset_path +
                                                   f'{self.dataset_type}_weighted_bow_vector_values.pickle')
        self.intersection_vocab_words_ids = self.weighted_vector_values_df.intersection_vocab_words_ids.to_list()
        self.bow_vector_components_values = self.weighted_vector_values_df.bow_vector_components_values.to_list()

    def preprocess_data(self):
        """
        Runs data preprocessing, vocab building and weighted vector values pre-computation.
        """
        self.preprocess_texts()
        if self.dataset_type == 'train':
            self.get_vocab()
        self.precompute_weighted_bow_vector_values()

    def preprocess_texts(self):
        """
        Preprocesses texts (applies punctuation and numbers removal, tokenization & stemming, stopwords removal).
        """
        nlp = spacy.load('en_core_web_sm')
        self.preprocessed_texts = []
        start_time, preprocessing_start_time = time.time(), time.time()
        stopwords = nlp.Defaults.stop_words
        for i, text in enumerate(self.texts):
            if i % 1e3 == 0:
                print(f'Preprocessed {i}/{self.texts_num} texts in {time.time() - start_time} sec')
                start_time = time.time()
            # punctuation and numbers removal
            clean_text = re.sub(r'[^A-Za-z]', ' ', text)
            # tokenization & stemming, stopwords removal
            stemmed_text = []
            for token in nlp(clean_text):
                stemmed_token = token.lemma_.strip().lower()
                if stemmed_token not in stopwords and len(stemmed_token) > 1:
                    stemmed_text.append(stemmed_token)
            self.preprocessed_texts.append(stemmed_text)
        print(f'Preprocessing time: {time.time() - preprocessing_start_time} sec')

        save_file(path=self.cfg.preprocessed_dataset_path + f'{self.dataset_type}_preprocessed.pickle',
                  columns_names=['text', 'preprocessed_text'],
                  columns_content=[self.texts, self.preprocessed_texts])

    def get_vocab(self):
        """
        Builds vocab.
        """
        top_words_sorted_by_freq = np.asarray(sorted(Counter(Counter(np.concatenate(self.preprocessed_texts))).items(),
                                                     key=lambda item: item[1], reverse=True)[:self.cfg.vocab_size])
        tokens, frequencies = top_words_sorted_by_freq[:, 0], top_words_sorted_by_freq[:, 1].astype(int)
        save_file(path=self.cfg.preprocessed_dataset_path + f'vocab.pickle',
                  columns_names=['token', 'frequency'],
                  columns_content=[tokens, frequencies])
        self.vocab = {k: v for k, v in zip(tokens, frequencies)}
        self.vocab_size = len(self.vocab)
        self.vocab_words = np.asarray(list(self.vocab.keys()))

    def precompute_weighted_bow_vector_values(self):
        """
        Precomputes weighted vector values and saves them and their locations (indexes in bow vector)
        for faster encoding while training.
        """
        self.intersection_vocab_words_ids, self.bow_vector_components_values = [], []
        start_time, preprocessing_start_time = time.time(), time.time()

        for i, text in enumerate(self.preprocessed_texts):
            if i % 1e2 == 0:
                print(f'Processed {i}/{len(self.preprocessed_texts)} texts in {time.time() - start_time} sec')
                start_time = time.time()

            current_text_words, current_text_words_counts = np.unique(text, return_counts=True)
            tf = np.log(1 + current_text_words_counts / len(text))
            idf = np.log(self.texts_num / np.asarray([self.vocab[term] if term in self.vocab_words else 1e-6
                                                      for term in current_text_words]))
            tf_idf = tf * idf
            _, intersection_current_text_words_ids, intersection_vocab_words_ids = np.intersect1d(
                current_text_words, self.vocab_words, assume_unique=True, return_indices=True)
            self.intersection_vocab_words_ids.append(intersection_vocab_words_ids)
            self.bow_vector_components_values.append(current_text_words_counts[intersection_current_text_words_ids] * \
                                                     tf_idf[intersection_current_text_words_ids])
        print(f'Preprocessing time: {time.time() - preprocessing_start_time} sec')

        save_file(self.cfg.preprocessed_dataset_path + f'{self.dataset_type}_weighted_bow_vector_values.pickle',
                  columns_names=['intersection_vocab_words_ids', 'bow_vector_components_values'],
                  columns_content=[self.intersection_vocab_words_ids, self.bow_vector_components_values])

    def get_encoded_text(self, idx):
        """
        Gets encoded text by idx with precomputed weighted values and their locations in bow vector.
        :param idx: text index in dataset
        :return: encoded text
        """
        encoded_text = np.zeros(self.vocab_size, dtype=np.float32)
        encoded_text[self.intersection_vocab_words_ids[idx]] = self.bow_vector_components_values[idx]
        return encoded_text

    def __len__(self):
        """
        Gets dataset length.
        :return: dataset length
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Gets dataset item (encoded text and corresponding label)
        :param idx: index for getting data
        :return: encoded_text and it`s label
        """
        encoded_text = self.get_encoded_text(idx)
        label = self.labels[idx]
        return encoded_text, label
