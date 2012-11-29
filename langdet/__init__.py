import random
from collections import defaultdict


def stream_sample(filename):
    with open(filename) as fin:
        for line in fin:
            lang, text = line[:-1].decode('utf8').split('\t')
            yield lang, text


class LanguageDetector(object):
    def preprocess(self, text):
        return text

    def train(self, samples):
        raise NotImplemented

    def detect(self, text):
        return 'xx', 0.0

    def eval(self, samples):
        tp = defaultdict(int)
        fn = defaultdict(int)
        fp = defaultdict(int)
        languages = set()

        mistakes = []
        for label, text in samples:
            languages.add(label)
            lang_code, _ = self.detect(text)

            if lang_code == label:
                tp[label] += 1
            else:
                mistakes.append((text, label, lang_code))
                fn[label] += 1
                fp[lang_code] += 1

        precision = {}
        recall = {}
        for lang in languages:
            if tp[lang] + fp[lang] == 0:
                precision[lang] = 0.0
            else:
                precision[lang] = tp[lang] / float(tp[lang] + fp[lang]) * 100.0

            if tp[lang] + fn[lang] == 0:
                recall[lang] = 0
            else:
                recall[lang] = tp[lang] / float(tp[lang] + fn[lang]) * 100.0

        return precision, recall, mistakes


class RandomLanguageDetector(LanguageDetector):
    def __init__(self):
        super(RandomLanguageDetector, self).__init__()
        self._random = random.Random()
        
    def train(self, samples):
        model = set()
        for label, _ in samples:
            model.add(label)
        model.add('xx')
        self._model = list(model)

    def detect(self, text):
        return self._random.choice(self._model), 1.0
