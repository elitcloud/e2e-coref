# ========================================================================
# Copyright 2018 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

# This is an adapter for https://github.com/kentonl/e2e-coref

import tensorflow as tf
from typing import Sequence

from elit.nlp.tokenizer import EnglishTokenizer
from elit.component import NLPComponent
from elit.structure import Document

import util
import coref_model

__author__ = "Liyan Xu"


class E2ECoref(NLPComponent):
    def __init__(self, experiment: str='final'):
        super(E2ECoref, self).__init__()

        self.tokenizer = EnglishTokenizer()
        self.config = util.initialize_experiment(experiment)
        self.model = coref_model.CorefModel(self.config)
        self.session = tf.Session()     # Currently no closing operation
        self.model.restore(self.session)

    def init(self, **kwargs):
        pass

    def load(self, model_path: str, **kwargs):
        pass

    def save(self, model_path: str, **kwargs):
        pass

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str, **kwargs) -> float:
        pass

    def evaluate(self, docs: Sequence[Document], **kwargs):
        pass

    def decode(self, docs: Sequence[Document], genre: str='nw', **kwargs) -> Sequence[Document]:
        for doc in docs:
            doc['sens'] = self.tokenizer.decode(doc['doc'])['sens']
            predicted = self.make_predictions(doc, self.model, self.session, genre)
            doc['coref'] = self.adapt_output(predicted)
        return docs

    def make_predictions(self, doc, model, session, genre: str):
        example = self.adapt_input(doc, genre)
        tensorized_example = model.tensorize_example(example, is_training=False)
        feed_dict = {i: t for i, t in zip(model.input_tensors, tensorized_example)}
        _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(
            model.predictions + [model.head_scores], feed_dict=feed_dict)

        predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

        example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends,
                                                                        predicted_antecedents)
        example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
        example["head_scores"] = head_scores.tolist()
        return example

    def adapt_input(self, doc: Document, genre: str):
        result = {
            "doc_key": genre,
            "clusters": [],
            "sentences": [],
            "speakers": [],
        }
        sentences = doc['sens']
        result['sentences'] = [sentence['tok'] for sentence in sentences]
        result['speakers'] = [['' for _ in sentence['tok']] for sentence in sentences]
        return result

    def adapt_output(self, orig_output, show_words=True):
        result = {
            "mention": [],
            "cluster": [],
        }
        words = util.flatten(orig_output["sentences"])

        result['mention'] = [(start, end) for start, end in orig_output['top_spans']]
        for cluster in orig_output['predicted_clusters']:
            curr = {'off': cluster}
            if show_words:
                curr['words'] = [" ".join(words[m[0]:m[1] + 1]) for m in cluster]
            result['cluster'].append(curr)
        return result


if __name__ == "__main__":
    sample_text = 'As a presidential candidate in August 2015, Donald Trump huddled with a longtime friend,' \
                  ' media executive David Pecker, in his cluttered 26th floor Trump Tower office and made a request.' \
                  ' What can you do to help my campaign? he asked, according to people familiar with the meeting.' \
                  ' Mr. Pecker, chief executive of American Media Inc., ' \
                  'offered to use his National Enquirer tabloid to buy the silence of women if they tried to ' \
                  'publicize alleged sexual encounters with Mr. Trump.'
    input_docs = [{'doc_id': 0, 'doc': sample_text}]
    coref = E2ECoref('test')
    print(coref.decode(input_docs))
