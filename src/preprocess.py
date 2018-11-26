from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

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
    def __init__(self, generator=True):
        self.pp = get_text_processor()
        self.generator = generator

    def pre_process_steps(self, X):
        for x in tqdm(X, desc="Preprocessing..."):
            yield self.pp.pre_process_doc(x)

    def transform(self, X, y=None):
        preprocessed = self.pre_process_steps(X)
        return preprocessed if self.generator else tuple(preprocessed)

    def fit(self, X, y=None):
        return self


if __name__ == '__main__':
    from sklearn.pipeline import Pipeline
    
    
    sentences = [
        "CANT WAIT for the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))",
        "I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/",
        "@SentimentSymp:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/."
    ]

    pipeline = Pipeline([('preprocess', PipelinePreprocessor(generator=False))])
    sentences = pipeline.fit_transform(sentences)
    
    for s in sentences:
        print(s)