import re
import math
import random
import unicodedata
from operator import itemgetter
from itertools import chain
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
        return list(text)

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


class BigramFeatureMixin(object):
    def _extract_features(self, text):
        return [text[i:i+2] for i in xrange(len(text)-1)]


class TrigramFeatureMixin(object):
    def _extract_features(self, text):
        return [text[i:i+3] for i in xrange(len(text)-2)]


class BigramCosineLanguageDetector(BigramFeatureMixin, CosineLanguageDetector):
    pass


class TrigramCosineLanguageDetector(TrigramFeatureMixin, CosineLanguageDetector):
    pass


word_tokenizer = re.compile('\w+', re.U).findall


class MultipleFeatureMixin(object):
    def _extract_features(self, text):
        text = unicodedata.normalize('NFC', text)
        text = ' '.join(word_tokenizer(text))
        unigrams = list(text)
        bigrams = [text[i:i+2] for i in xrange(len(text)-1)]
        trigrams = [text[i:i+3] for i in xrange(len(text)-2)]
        return [x for x in chain(unigrams, bigrams, trigrams) if not x.isdigit()]

    def _normalize_vector(self, v):
        norm = [0.0, 0.0, 0.0]
        for k, x in v.iteritems():
            norm[len(k)-1] += x*x

        for i in range(len(norm)):
            norm[i] = math.sqrt(norm[i])

        for k in v:
            v[k] /= norm[len(k)-1]


class MultiCosineLanguageDetector(MultipleFeatureMixin, CosineLanguageDetector):
    pass
