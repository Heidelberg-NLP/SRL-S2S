import os, sys
from logging import getLogger, StreamHandler, DEBUG, FileHandler, Formatter

def set_log(fname):
    dirname = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(dirname, fname)

    # ログの出力名を設定
    logger = getLogger(__name__)

    # コンソール出力するための設定
    handler = StreamHandler()
    logger.addHandler(handler)

    # ログのファイル出力先の設定
    #if os.path.isfile(fname):
    #    os.remove(fname)
    flog = FileHandler(fname, mode="a") # default mode = "a"
    logger.addHandler(flog)

    # level
    """
    NOTSET:     設定値などの記録
    DEBUG:      動作確認などのデバッグ記録
    INFO:       正常動作の記録
    WARNING:    ログの定義名
    ERROR:      エラーなどの重大な問題
    CRITICAL:   停止などの致命的な問題
    """
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)

    # ログの出力形式の設定
    formatter = Formatter('%(asctime)s - line:%(lineno)d - __%(levelname)s__ - %(message)s')
    flog.setFormatter(formatter)
    handler.setFormatter(formatter)

    logger.propagate = False
    return logger

