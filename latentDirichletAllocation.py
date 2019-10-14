import numpy as np
from tqdm import trange, tqdm
from scipy.special import polygamma, gamma


class LDA:

    def __init__(self, wordIndex, txtIndex, txt2Word,
                 topicnum, iterate, e_iterate, m_iterate):

        self.__iterate = iterate  # 全体のイテレーション回数
        # self.__eIterate = e_iterate  # E-stepでのイテレーション回数
        self.__mIterate = m_iterate  # M-stepでのイテレーション回数
        self.__txt2word = txt2Word
        self.__txtIndex = txtIndex
        self.__wordIndex = wordIndex
        self.__topicnum = topicnum

        self.__alpha = np.ones(self.__topicnum) / self.__topicnum
        # α全体をトピック数の逆数で初期化
        self.__beta = np.random.rand(self.__topicnum, len(self.__wordIndex))
        self.__beta = self.__beta / np.sum(self.__beta, axis=1).reshape(-1, 1)
        self.__beta.clip(min=0.000000000001)
        # βを乱数で初期化し、制約条件を満たすように計算

        self.__gamma = np.zeros((len(self.__txtIndex), self.__topicnum))
        # γは後々計算
        self.__phi = ([[] for i in range(len(self.__txtIndex))])  # φも同様
        for t in self.__txtIndex.keys():
            i = self.__txtIndex[t]
            self.__phi[i] = np.zeros(
                (len(self.__txt2word[t]), self.__topicnum))

        for t in self.__txt2word.keys():
            d = self.__txtIndex[t]
            wordList = len(self.__txt2word[t])
            # φの初期化
            self.__phi[d][:, :] = 1 / self.__topicnum

            # γの初期化
            self.__gamma[d] = self.__alpha + (wordList / self.__topicnum)
            self.__gamma[d] = self.__gamma[d].clip(min=0.00000000001)

    def __e_step(self):

        # E-step
        # 文書ごとに処理を行っていく
        for t in tqdm(self.__txt2word.keys(), desc='E-step Iterate',
                      leave=False):
            d = self.__txtIndex[t]
            w_n = [self.__wordIndex[w] for w in self.__txt2word[t]]

            # 更新処理
            self.__phi[d] = self.__beta[:, w_n].T * \
                np.exp(polygamma(0, self.__gamma[d]))
            # 制約条件
            self.__phi[d] /= np.sum(self.__phi[d])
            self.__gamma[d] = self.__alpha + np.sum(self.__phi[d], axis=0)
            self.__phi[d] = self.__phi[d].clip(min=0.0000000000001)
            self.__gamma[d] = self.__gamma[d].clip(min=0.00000000001)

    def __m_step(self):

        # β更新
        self.__beta = np.zeros((self.__topicnum, len(self.__wordIndex)))
        for t in tqdm(self.__txtIndex.keys(), desc='M-step β Update',
                      leave=False):
            d = self.__txtIndex[t]
            j = [self.__wordIndex[w] for w in self.__txt2word[t]]
            self.__beta[:, j] += self.__phi[d].T
        # 制約条件
        self.__beta = self.__beta / np.sum(self.__beta, axis=1).reshape(-1, 1)
        self.__beta = self.__beta.clip(min=0)

        # α更新
        M = len(self.__txtIndex)
        for m in trange(self.__mIterate, desc='M-step α Iterate',
                        leave=False):
            h = -M * polygamma(1, self.__alpha)
            z = -M * polygamma(1, np.sum(self.__alpha))
            g = M * (polygamma(0, np.sum(self.__alpha)) -
                     polygamma(0, self.__alpha)) + \
                np.sum(polygamma(0, self.__gamma) -
                       polygamma(0, np.sum(self.__gamma)))
            c = np.sum(g / h) / (z**(-1) + np.sum(h**(-1)))
            self.__alpha = self.__alpha - ((g - c) / h)
            self.__alpha = self.__alpha.clip(min=0.00000000001)

    def learning(self):

        # with np.errstate(invalid='ignore'):
        iterate = trange(self.__iterate, desc='Likehood: ?')
        for i in iterate:
            if i % 10 == 0:
                iterate.set_description(
                    'Likehood {}'.format(self.__likehood())
                )
            self.__e_step()
            self.__m_step()

    def __likehood(self):
        L = 0.0
        for t in tqdm(self.__txt2word.keys(), leave=False):
            d = self.__txtIndex[t]
            j = [self.__wordIndex[w] for w in self.__txt2word[t]]
            E_theta_p = np.log(gamma(np.sum(self.__alpha))) - \
                np.sum(np.log(gamma(self.__alpha))) + \
                np.sum(
                (self.__alpha - 1) * (polygamma(0, self.__gamma[d]) -
                                      polygamma(0, np.sum(self.__gamma)))
            )
            E_z_p = np.sum(np.sum(
                self.__phi[d] * (polygamma(0, self.__gamma[d]) -
                                 polygamma(0, np.sum(self.__gamma[d]))),
            )
            )
            E_w_p = np.sum(np.sum(self.__phi[d] * np.log(self.__beta[:, j].T)))

            E_theta_q = np.log(gamma(np.sum(self.__gamma[d]))) - \
                np.sum(np.log(gamma(self.__gamma[d]))) + \
                np.sum((self.__gamma[d] - 1) * (polygamma(0, self.__gamma[d]) -
                                                polygamma(0, np.sum(self.__gamma[d])))
                       )
            E_z_q = np.sum(
                np.sum(self.__phi[d] * np.log(self.__phi[d]), axis=1))

            L += E_theta_p + E_z_p + E_w_p - E_theta_q - E_z_q

        return L

    def getTopicWord(self, topic, topnum):
        topwords_in_topic = [k for k, i in self.__wordIndex.items()
                             if i in np.argsort(self.__beta[topic])[::-1][:topnum]]
        topwords_with_prob = {w: self.__beta[topic][self.__wordIndex[w]]
                              for w in topwords_in_topic}
        return topwords_with_prob

    def getBookTopic(self, bookname):
        book_topics = self.__gamma[self.__txtIndex[bookname]]
        return book_topics
