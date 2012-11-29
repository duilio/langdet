import random
import math
from operator import itemgetter
from collections import defaultdict


def stream_sample(filename):
    with open(filename) as fin:
        for line in fin:
            lang, text = line[:-1].decode('utf8').split('\t')
            yield lang, text


class LanguageDetector(object):
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
    def train(self, samples):
        model = set()
        for label, _ in samples:
            model.add(label)
        model.add('xx')
        self._model = list(model)

    def detect(self, text):
        return random.choice(self._model), 1.0


class CosineLanguageDetector(LanguageDetector):
    def _extract_features(self, text):
        return map(''.join, zip(text, text[1:]))

    def _normalize_vector(self, v):
        norm = math.sqrt(sum(x*x for x in v.itervalues()))
        for k in v:
            v[k] /= norm
        
    def train(self, samples):
        extract_features = self._extract_features

        model = defaultdict(lambda: defaultdict(float))
        for label, text in samples:
            features = extract_features(text)
            for f in features:
                model[label][f] += 1

        for v in model.itervalues():
            self._normalize_vector(v)

        self._model = model

    def detect(self, text):
        features = self._extract_features(text)
        u = defaultdict(float)
        for f in features:
            u[f] += 1
        self._normalize_vector(u)

        r = []
        for l, v in self._model.iteritems():
            score = 0.0
            for f in u:
                score += u[f] * v.get(f, 0.0)
            r.append((l, score))
        return max(r, key=itemgetter(1))
