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
import sys
from bisect import bisect_right

from elit.component import NLPComponent
from elit.structure import Document

import util
import coref_model

__author__ = "Liyan Xu"


class E2ECoref(NLPComponent):
    def __init__(self, experiment: str='final', path_context_emb: str=None, path_head_emb: str=None,
                 dir_elmo: str=None, dir_log_root: str=None, path_char_vocab: str=None):
        '''
        :param experiment: 'final' or 'test'
        :param path_context_emb: absolute path of context embedding
        :param path_head_emb: absolute path of head embedding
        :param dir_elmo: absolute path of elmo directory
        :param dir_log_root: absolute path of log root directory
        :param path_char_vocab: absolute path of char_vocab file
        '''
        super(E2ECoref, self).__init__()

        self.config = util.initialize_experiment(
            experiment, path_context_emb, path_head_emb, dir_elmo, dir_log_root, path_char_vocab)
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

    def decode(self, docs: Sequence[Document], genre: str='nw', show_words=False, **kwargs) -> Sequence[Document]:
        for doc in docs:
            predicted = self.make_predictions(doc['sens'], self.model, self.session, genre)
            if show_words:
                doc['coref'], doc['coref_words'] = self.adapt_output(predicted, show_words)
            else:
                doc['coref'] = self.adapt_output(predicted, show_words)
        return docs

    def make_predictions(self, sentences, model, session, genre: str):
        example = self.adapt_input(sentences, genre)
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

    def adapt_input(self, sentences, genre: str):
        return {
            "doc_key": genre,
            "clusters": [],
            "sentences": [sentence['tok'] for sentence in sentences],
            "speakers": [['' for _ in sentence['tok']] for sentence in sentences],
        }

    def adapt_output(self, orig_output, show_words):
        sen_start_idx = []  # starting index of each sentence
        curr_start_idx = 0
        for sen in orig_output["sentences"]:
            sen_start_idx.append(curr_start_idx)
            curr_start_idx += len(sen)

        # (tok_off_global, tok_off_global) -> (sen_idx, tok_off_sen, tok_off_sen)
        def transform_off(off, sen_idx):
            return [sen_idx, off[0] - sen_start_idx[sen_idx], off[1] - sen_start_idx[sen_idx] + 1]

        coref = [[transform_off(off, bisect_right(sen_start_idx, off[0]) - 1) for off in cluster]
                 for cluster in orig_output['predicted_clusters']]

        if not show_words:
            return coref
        else:
            coref_words = [[" ".join(orig_output["sentences"][off[0]][off[1]:off[2]]) for off in cluster] for cluster in coref]
            return coref, coref_words


if __name__ == "__main__":
    from elit.nlp.tokenizer import EnglishTokenizer
    sample_text = 'As a presidential candidate in August 2015, Donald Trump huddled with a longtime friend,' \
                  ' media executive David Pecker, in his cluttered 26th floor Trump Tower office and made a request.' \
                  ' What can you do to help my campaign? he asked, according to people familiar with the meeting.' \
                  ' Mr. Pecker, chief executive of American Media Inc., ' \
                  'offered to use his National Enquirer tabloid to buy the silence of women if they tried to ' \
                  'publicize alleged sexual encounters with Mr. Trump.'
    input_docs = [EnglishTokenizer().decode(sample_text)]
    experiment = 'test' if len(sys.argv) == 1 else sys.argv[1]
    coref = E2ECoref(experiment)
    print(coref.decode(input_docs, show_words=True))
