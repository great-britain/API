import pandas as pd
import numpy as np
import torch
import re

import warnings

warnings.filterwarnings("ignore")

from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from sentence_transformers import SentenceTransformer, util


class StroyModel:
    def __init__(self,
                 ksr_file: str,
                 train_file: str,
                 # model: SentenceTransformer,
                 # embeddings: torch.Tensor,
                 model_name: str = 'intfloat/multilingual-e5-small',
                 k_len: int = 10):
        self.k_len = k_len
        self.ksr_df = self.load_csr(ksr_file)
        self.train_df = self.load_train(train_file)

        # обрезаем маленькие коды
        self.preproc_ksr()
        self.texts = self.create_texts()

        # загружаем BM25
        self.bm25_retriever = self.load_BM25()

        # Обрабатываем наименования
        self.preproc_titles()

        # Загружаем модель
        # self.model = model
        # self.embeddings_KSR = embeddings
        self.load_model(model_name=model_name)

    def load_model(self, model_name='intfloat/multilingual-e5-large'):
        """
        Загружаем модель
        """
        print(f'Загружаем модель ...')
        self.model = SentenceTransformer(model_name)  # 0.085 8min 0.075
        self.embeddings_KSR = self.model.encode(self.ksr_df['Наименование'].values, convert_to_tensor=True)
        print(f'Загрузили модель!')

    def load_csr(self, ksr_file):
        # КСР предоставлен кейсодержателем
        ksr_df = pd.read_excel(ksr_file, header=1, engine='openpyxl')
        ksr_df = ksr_df.dropna()
        for i in ['Книга', 'Часть', 'Раздел', 'Группа']:
            ksr_df[i] = ksr_df['Код ресурса'].map(lambda x: x.split(':')[0] if i in x else np.nan)
            ksr_df[i] = ksr_df[i].fillna(method='ffill')
        ksr_df = ksr_df.dropna()
        ksr_df = ksr_df[~(ksr_df['Код ресурса'] == ksr_df['Наименование'])]
        ksr_df = ksr_df[["Книга", "Часть", "Раздел", "Группа", "Код ресурса", "Наименование", "Ед.изм."]]
        return ksr_df

    def load_train(self, train_file):
        return pd.read_excel(train_file, header=0, engine='openpyxl')

    def load_BM25(self):
        """
        загружаем BM25
        """
        bm25_retriever = BM25Retriever.from_documents(self.texts)
        bm25_retriever.k = self.k_len
        return bm25_retriever

    def preproc_ksr(self):
        # Из классификатора заведомо исключены верхнеуровневые коды (короткие), чтобы они не попадали в ответы
        self.ksr_df['len_code'] = self.ksr_df['Код ресурса'].apply(lambda x: len(x.strip()))
        self.ksr_df = self.ksr_df[self.ksr_df['len_code'] >= 24]

    def create_texts(self):
        # грузим фрейм в лоадер, выделив колонку для векторизации
        loader = DataFrameLoader(self.ksr_df, page_content_column='Наименование')
        documents = loader.load()
        # нарезаем документы на одинаковые чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )  # chunk_size - размер чанки на которую
        texts = text_splitter.split_documents(documents)
        return texts

    def preproc_text(self, text):
        result_text = text
        result_text = re.sub(r'[\.\,\'"=]', ' ', result_text)
        result_text = re.sub(r'[-)(\\\/]', ' ', result_text)
        result_text = re.sub(r'([\d\.]+)\s*?[xхХX\\\/]\s*?([\d\.]+)', r'\1 \2 ', result_text)
        result_text = re.sub(r'[^А-Яа-я0-9A-Za-z,\.\sx]', ' ', result_text)
        result_text = re.sub(r'\s+', ' ', result_text)
        result_text = result_text.lower()
        return result_text

    #     text = 'Штанги анкерные инъекционные (для бурения) 103/78, длина, м 3'
    #     preproc_text(text)

    def preproc_titles(self):
        """
        Обрабатываем наименования
        """
        self.ksr_df['preproc_name'] = self.ksr_df['Наименование'].apply(lambda x: self.preproc_text(x))
        self.train_df['preproc_name'] = self.train_df['record_name'].apply(lambda x: self.preproc_text(x))

    def predict_code_bm25(self, query_text):
        docs = self.bm25_retriever.get_relevant_documents(query_text)
        return [doc.metadata['Код ресурса'] for doc in docs]

    def predict_code_llm(self, query_text, k):
        embedding_query = self.model.encode(query_text, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(embedding_query, self.embeddings_KSR)
        top_results = torch.topk(cos_scores, k=k)
        result = self.ksr_df.iloc[top_results.indices.tolist()[0]][['Наименование', 'Код ресурса']]
        return list(result['Код ресурса'].values), top_results.values.tolist()[0]

    def calc_score_codes(self, query_text):
        ksr_codes_llm, scores = self.predict_code_llm(query_text, k=self.k_len)
        ksr_codes_bm25 = self.predict_code_bm25(query_text)

        count_codes = max(len(ksr_codes_bm25), len(ksr_codes_llm))
        score_codes = {}
        # Рейтинг будем считать как в соревнованиях перемножение мест
        for index, bm25_code in enumerate(ksr_codes_bm25):
            score_codes[bm25_code] = {}
            score_codes[bm25_code]['place'] = index + 1
            # score_codes[bm25_code]['place_1'] = index + 1
            # Если у кода нет скора по косинусному расстоянию, тогда указываем минимальное
            if bm25_code not in ksr_codes_llm:
                score_codes[bm25_code]['score'] = min(scores)
                # score_codes[bm25_code]['place_2'] = count_codes
                score_codes[bm25_code]['place'] = score_codes[bm25_code]['place'] * count_codes

        for index, llm_code in enumerate(ksr_codes_llm):
            if llm_code in score_codes:
                score_codes[llm_code]['place'] = score_codes[llm_code]['place'] * (index + 1)
                # score_codes[llm_code]['place_2'] = index + 1
            else:
                score_codes[llm_code] = {}
                score_codes[llm_code]['place'] = index + 1
                # score_codes[llm_code]['place_2'] = index + 1
                # score_codes[llm_code]['place_1'] = count_codes
                score_codes[llm_code]['place'] = score_codes[llm_code]['place'] * count_codes

            score_codes[llm_code]['score'] = scores[index]
        return score_codes, count_codes

    def predict_submit(self, query_text):
        score_codes, count_codes = self.calc_score_codes(query_text)

        current_code = None
        current_code_score = count_codes * count_codes
        for ksr_code in score_codes:
            final_score = score_codes[ksr_code]['place'] / score_codes[ksr_code]['score']
            score_codes[ksr_code]['final_score'] = final_score
            if final_score < current_code_score:
                current_code = ksr_code
                current_code_score = final_score
        return current_code

    def predict_API(self, query_text):
        score_codes, _ = self.calc_score_codes(query_text)

        for ksr_code in score_codes:
            final_score = score_codes[ksr_code]['place'] / score_codes[ksr_code]['score']
            score_codes[ksr_code]['final_score'] = final_score
        ksr_code, cos_sim, final_score = \
            pd.DataFrame.from_dict(score_codes, orient='index').sort_values(by='final_score').reset_index().iloc[0][
                ['index', 'score', 'final_score']].values
        ksr_name = self.ksr_df[self.ksr_df['Код ресурса'] == ksr_code]['Наименование'].item()


        return ksr_code, ksr_name, cos_sim, final_score


if __name__ == '__main__':
    PATH = '.'
    DATASET_PATH = PATH + '/datasets'
    ksr_file = DATASET_PATH + '/classification.xlsx'
    train_file = DATASET_PATH + '/train.xlsx'

    stroy_model = StroyModel(
        ksr_file=ksr_file,
        train_file=train_file,
        model_name='intfloat/multilingual-e5-small',
        k_len=50)

    query_text = 'Анкер забивной М10 DRM 12x40 сталь'
    print(stroy_model.predict_submit(query_text))
    print(stroy_model.predict_API(query_text))
