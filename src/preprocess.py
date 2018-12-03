from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import numpy as np

# custom imports
try:
    from emojis_dict import emojis
except ImportError:
    from .emojis_dict import emojis


def get_text_processor():
    text_processor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
            'time', 'date', 'number'],
        annotate={"hashtag", "allcaps", "elongated", "repeated",
            'emphasis', 'censored'},
        fix_html=True,
        segmenter="twitter", 
        corrector="twitter", 
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emojis, emoticons]
    )
    
    return text_processor


class PipelinePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pp = get_text_processor()

    def pre_process_steps(self, X):
        tqdm.pandas(desc="Preprocessing...")
        return X.progress_apply(self.pp.pre_process_doc)

    def transform(self, X, y=None):
        return self.pre_process_steps(X)

    def fit(self, X, y=None):
        return self


if __name__ == '__main__':
    from sklearn.pipeline import Pipeline
    
    
    sentences = [
        "CANT WAIT for the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))",
        "I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/",
        "@SentimentSymp:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/."
    ]

    pipeline = Pipeline([('preprocess', PipelinePreprocessor())])
    sentences = pipeline.fit_transform(sentences)
    
    for s in sentences:
        print(s)