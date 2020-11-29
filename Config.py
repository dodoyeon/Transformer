import json

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod # python 정적 메소드 -> self.대신 cls.라는 인자를 가진다.
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.load(f.read())
            return Config(config)

config = Config.load('Config.json') # config 파일은 train에서만 사용하구
                                    # 깊은 코딩부분(모델부분)에서는 쓰지않는다!!!