__doc__ = """
    OverView:
    テキストファイル、もしくはそれらが入ったディレクトリを受取り、形態素解析の結果をcsvに出力する

    Usage:
        parsingTxt <target> <output> [-r | --directroy] [-x <altPos>] [-d | --dict <dictionary_path>] [-p  <pos>] 

    Options:
        target                           : 解析対象のtxtファイル、もしくはディレクトリ
        output                           : 出力先のcsv
        -r --directroy                   : 対象がディレクトリの場合は設定必須
        -x <altPos>                      : 品詞推定の切り替え(on/off)及び、未知語の出力品詞の設定
        -d --dict <dictionary_path>      : 外部辞書を使用する場合
        -p <pos>                         : 抽出対象の形態素の品詞
"""

import MeCab
import os
from mimetypes import guess_type
from docopt import docopt


def main():
    args = docopt(__doc__)

    target = args['target']
    output = args['output']

    # NOTE: mecab の初期化
    initCommand = ''
    if len(args['-x']) > 0:
        initCommand += '-x'
        initCommand += args['-x']

    if len(args['dictionary_path']) > 0:
        initCommand += '-d'
        if os.path.isdir(args['dictionary_path']):
            initCommand += args['dictionary_path']
        else:
            raise FileNotFoundError 
    m = MeCab.Tagger(initCommand)

    if len(target) > 0:
        if args['--directroy'] is True and \
                os.path.isdir(target):
            txtList = os.listdir(target)

        elif os.path.isfile(target):
            txtList = [target]  # 配列として渡して後の処理の記述を一括化

    for tname in txtList:
        with open(target, mode='r') as txt:
            rawResult = m.parse(txt.readline())

    else:
        raise Exception('Please Input targetPath')
