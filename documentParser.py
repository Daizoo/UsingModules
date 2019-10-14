import MeCab
import argparse
import os

# -----CLから直接呼び出す場合での設定
# 引数のparserの設定
parser = argparse.ArgumentParser(
    description='MeCabでの形態素解析を文書一括で行ってついでに語彙一覧や文書や単語のインデキシングもする' +
    '\n結果は全てjsonで保存される'
)
# 必須引数
parser.add_argument('-t', '--targetDir',
                    help='解析対象の文書が入ったディレクトリ', required=True)
parser.add_argument('-s', '--saveDir',
                    help='解析結果の保存先ディレクトリ', default='./parseResult')
# オプション引数
parser.add_argument('-m', '--morph', nargs='*', help='取得したい形態素の品詞')
parser.add_argument('-d', '--dict', help='形態素解析に使う辞書へのパス', default='')
parser.add_argument('-e', '--estimate', action='store_true', help='品詞推定機能を付ける')
parser.add_argument('-f', '--features', help='解析結果から何番目のデータを取り出すか',
                    type=int, default=-1)
# 設定ここまで


def documentParser(txtList, dirPath,
                   morphList=[], dictPath='', target=-1, estimate=False):
    """
    MeCabによる文書一括形態素解析関数
    ついでに文書と形態素をインデキシングもして語彙も作成する

    Parameters
    -----------
    txtList : list of string
        解析対象の文書ファイル名一覧。
    dirPath : string
        解析対象の文書のあるディレクトリ
    morphList : list of string, default []
        取得したい形態素の品詞一覧、なくてもOK
    dictPath : string, default ''
        解析に使う辞書のディレクトリへのパス
    target : int, default 0
        形態素として格納したいデータの要素番号の指定。
        0の場合、表層形を格納する
    estimate : boolean, default False
        MeCabの品詞推定機能を使うかの指定
        Trueで使用する。
        未使用の場合、未知語として出力

    Returns
    --------
    vocab : set
        語彙群
    word2Index : dict
        形態素をキーとして対応する番号が格納されている
    txt2Index : dict
        文書名をキーとして対応する番号が格納されている
    txt2Word : dict
        文書名をキーとしてその文書から得られた形態素群が格納されている
    """

    option = ''
    vocab = set()
    word2Index = {}
    txt2Index = {}
    txt2Word = {}

    if estimate is not True:
        option += '-x 未知後'

    if len(dictPath) > 0:
        option = option + ' -d ' + dictPath

    m = MeCab.Tagger(option)

    for num, tname in enumerate(txtList):

        docWordList = []
        txt2Index[tname] = num
        with open(dirPath + '/' + tname) as txt:
            rawParse = m.parse(txt.read()).split('\n')

        for p in rawParse:
            results = p.split('\t')
            if len(results) < 2:
                break

            features = results[1].split(',')
            pos = features[0]
            if target >= 0:
                if target >= len(features):
                    word = results[0]
                else:
                    word = features[target]
            else:
                word = results[0]

            if len(morphList) > 0 and pos in morphList:
                docWordList.append(word)
                vocab.add(word)
            elif len(morphList) <= 0:
                docWordList.append(word)
                vocab.add(word)

        txt2Word[tname] = docWordList

    for num, v in enumerate(list(vocab)):
        word2Index[v] = num

    return word2Index, txt2Index, txt2Word


if __name__ == '__main__':
    import json

    args = parser.parse_args()
    targetDir = args.targetDir
    saveDir = args.saveDir
    if os.path.isdir(saveDir) is False:
        os.mkdir(saveDir)

    if args.morph is not None:
        morph = args.morph
    else:
        morph = []
    dict = args.dict
    estimate = args.estimate
    feature = args.features

    txtList = []
    for fname in os.listdir(targetDir):
        base, ext = os.path.splitext(fname)
        if ext == '.txt':
            print(fname)
            txtList.append(fname)

    w2i, t2i, t2w = documentParser(txtList=txtList, dirPath=targetDir,
                                   morphList=morph, dictPath=dict,
                                   target=feature, estimate=estimate)

    with open(saveDir + '/word2index.json', 'wt') as j:
        json.dump(w2i, j, indent=4, ensure_ascii=False)

    with open(saveDir + '/txt2index.json', 'wt') as j:
        json.dump(t2i, j, indent=4, ensure_ascii=False)

    with open(saveDir + '/txt2word.json', 'wt') as j:
        json.dump(t2w, j, indent=4, ensure_ascii=False)
