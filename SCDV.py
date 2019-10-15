import json

import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.mixture.gaussian_mixture import GaussianMixture as gmm


class scdv:
    def __init__(self, txt2word_path):
        self.__txt2word = json.load(open(txt2word_path))
        self.__corpus = []
        self.__txtindex = {}
        self.__vocab = set()
        self.__D = len(self.__txt2word)

        for num, book in enumerate(self.__txt2word.items()):
            self.__txtindex[book[0]] = num
            self.__corpus.append(book[1])
            self.__vocab.update(book[1])

    def calc_docvec(self):
        wcv, wcv_idx = self.createVec()
        docv = np.zeros((self.__D, 300 * 9))
        a_max = 0
        a_min = 0
        for title, words in self.__txt2word.items():
            i = self.__txtindex[title]
            for w in words:
                docv[i] += wcv[wcv_idx[w]]

            docv[i] /= len(words)
            docv[i] /= np.linalg.norm(docv[i])
            a_max += docv[i].max()
            a_min += docv[i].min()

        a_max /= self.__D
        a_min /= self.__D
        p = 4
        t = (p / 100) * ((np.abs(a_max) + np.abs(a_min)) / 2)
        docv[abs(docv) < t] = 0

        return docv, self.__txtindex

    def createVec(self):
        wcv = np.zeros((len(self.__vocab), 300 * 9))
        wcv_idx = {}
        w2v = Word2Vec(self.__corpus, size=300, min_count=0)
        word_vectors = w2v.wv.vectors
        clus = gmm(n_components=9, max_iter=100)
        clus.fit(word_vectors)
        gmm_idx_proba = clus.predict_proba(word_vectors)
        idf_idx, idf_vec = self.idfCalc()
        for num, word in enumerate(list(self.__vocab)):
            wcv_idx[word] = num
            wv = word_vectors[w2v.wv.vocab[word].index]
            probas = gmm_idx_proba[w2v.wv.vocab[word].index]
            for i in range(0, 9):
                wcv[num][i * 300 :(i + 1) * 300] = (
                    wv * probas[i] * idf_vec[idf_idx[word]]
                )

        return wcv, wcv_idx

    def idfCalc(self):
        idf_index = {}
        idf_vec = []
        for num, word in enumerate(list(self.__vocab)):
            idf_index[word] = num
            count = 0
            for b in self.__txt2word.values():
                if word in b:
                    count += 1
            idf_vec.append(count)

        idf_vec = np.array(idf_vec, dtype=np.float32)
        idf_vec /= self.__D
        idf_vec = np.log10(idf_vec ** -1)
        return idf_index, idf_vec


if __name__ == "__main__":
    test = scdv("./data/book_full_wakachi/txt2word.json")
    result = test.calc_docvec()

