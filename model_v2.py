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
                 # model:SentenceTransformer,
                 # embeddings:torch.Tensor,
                 model_name: str = 'intfloat/multilingual-e5-small',
                 k_len: int = 10,
                 page_content_column_bm25: str = 'preproc_name',
                 page_content_column_llm: str = 'Наименование'
                 ):

        # self.page_content_column = page_content_column
        self.page_content_column_llm = page_content_column_llm
        self.page_content_column_bm25 = page_content_column_bm25

        # Словари синонимов и брендов
        self.brands = ['klester', 'ceresit', 'bergauf', 'violgrunt standart', 'archin', 'ceresit', 'поливент супер',
                       'технониколь', 'makita', 'espira', 'milwaukee', 'практика', 'акустик-стандарт', 'universal',
                       'сталер', 'октава', 'bolleto', 'bergauf', 'klester', 'огнет', 'ntanу', 'электропайп', 'хемкор',
                       'агригазполимер', 'брэкс', 'wago', 'duplex', 'bettermann', 'avclink', 'flexicore', 'hyperline',
                       'nikolan', 'кедр', 'конкорд', 'калининград', 'кольчугино', 'tasker', 'krass', 'красс', 'термо',
                       'thermex', 'varteg', 'prime', 'аристон', 'aguasfera', 'firat', 'fdplast', 'valfex', 'optima',
                       'kleber', 'огнебазальт', 'basic', 'tech-krep', 'церезит', 'farbitex', 'ростурпласт', 'bitumast',
                       'партнер', 'zandz', 'лан юнион', 'сегментлан', 'альгиз', 'алюр', 'unis', 'юнис-плюс', 'perfekta',
                       'plitonit', 'danfoss', 'fiber', 'hykol', 'datafiber', 'подольсккабель', 'prysmian', 'generica',
                       'frlsltx', 'гидроконтур', 'аквастоп', 'ветонит', 'дауер', 'основит', 'masteremaco', 'синикон',
                       'кнауф', 'gross', 'alfa', ]
        self.synonyms = {'грунтовка': 'краска', 'штукатурка': 'смеси', 'уолок': 'профиль', 'вентиль': 'клапан',
                         'анкер-болт': 'болты', 'гайка': 'болты', 'анкерный элемент': 'анкер', 'известняк': 'камни',
                         'анкерный стержень': 'анкер', 'анкерный болт': 'анкер', 'бокс': 'шкаф', 'болт': 'анкер',
                         'аэратор': 'аэратор', 'минеральная вата': 'плиты из минеральной ваты', 'смесь': 'база',
                         'штукатурка': 'мастика', 'геотекстиль': 'геполотно', 'прибор': 'коробки', 'шпонка': 'смеси',
                         'гипсокартон': 'листы', 'грунт': 'грунтовка', 'грунт-эмаль': 'грунтовка', 'двери': 'блок',
                         'ручки': 'ручка-завертка', 'держатель': 'анкер', 'держатель': 'аэратор',
                         'полотно': 'геотекстиль', 'сетка': 'геосетка', 'полотно': 'геополотно',
                         'дроссель': 'дроссель-клапан', 'дроссель-клапан': 'клапан', 'дорожные плиты': 'делиниатор',
                         'дюбель': 'зажим', 'дюбель': 'дюбель', 'дюбель': 'зажимы', 'дюбель': 'хомуты',
                         'анкер': 'капсулы', 'анкерный элемент': 'анкер',
                         'изделия теплоизоляционные': 'плиты теплоизоляционные', 'кабель': 'зажим', 'гранит': 'плитка',
                         'раствор': 'смеси', 'клей': 'состав', 'клемма': 'зажим', 'колодка': 'кабель',
                         'комплект': 'бирки', 'хомут': 'гайки', 'корпус': 'коробка', 'ремонтный состав': 'смеси',
                         'лестница': 'блок', 'лестница': 'блок', 'манжета': 'комплект', 'маты': 'подложка',
                         'рулон': 'материал', 'накладка': 'блок', 'провод': 'зажимы', 'обводное колено': 'муфта',
                         'огнезащитное покрытие': 'материал', 'пробка-заглушка': 'заглушка',
                         'тепловая пушка': 'вентилятор', 'пленка': 'воск', 'пенополистирол': 'плиты',
                         'пиломатериал': 'доска', 'пирамида': 'смеси', 'пласт': 'шкаф', 'пластина': 'шайба',
                         'пленка': 'материал', 'плитка': 'брусчатка', 'лист': 'сталь', 'провод': 'зажимы',
                         'противопожарное полотно': 'материал', 'смола': 'клей', 'рамка': 'коробки', 'редуктор': 'кран',
                         'рулон': 'сталь', 'саморезы': 'воронка', 'сегмент': 'кабель', 'шайба': 'гайки',
                         'пушка': 'вентилятор', 'уплоитель': 'мастика', 'праймер': 'мастика', 'техноэласт': 'материал',
                         'рейка': 'скоба', 'держатель кровельный': 'заглушка', 'фанера': 'плиты', 'наконечник': 'болты',
                         'цемент': 'портландцемент', 'шайба': 'гайки', 'шпаклевка': 'база', 'шпилька': 'анкер',
                         'шумоглушитель': 'глушитель', 'щит': 'кабель', 'щув': 'кабель', 'крышка': 'загулшка',
                         'экранированная витая пара': 'кабель', 'клей': 'смеси', 'пруток-катанка': 'катанка',
                         'тройник/отвод': 'тройники', 'дюбель': 'шурупы', 'саморез': 'шурупы', 'выкл.': 'выключатели',
                         'электрокабель': 'кабель', 'подольсккабель': 'кабель', 'камкабель': 'кабель'}

        self.k_len = k_len
        self.ksr_df = self.load_csr(ksr_file)
        self.train_df = self.load_train(train_file)

        # обрезаем маленькие коды
        self.preproc_ksr()
        # Обрабатываем наименования
        self.preproc_titles()
        # Формируем документы на основе текстов
        self.texts = self.create_texts()

        # загружаем BM25
        self.bm25_retriever = self.load_BM25()

        # Загружаем модель
        # self.model = model
        # self.embeddings_KSR = embeddings
        self.load_model(model_name=model_name)

        self.vocab_ksr = self.get_vocab_ksr()

    def load_model(self, model_name: str):
        """
        Загружаем модель
        """
        print(f'Загружаем модель ...')
        self.model = SentenceTransformer(model_name)  # 0.085 8min 0.075
        self.embeddings_KSR = self.model.encode(self.ksr_df[self.page_content_column_llm].values,
                                                convert_to_tensor=True)
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
        train_df = pd.read_excel(train_file, header=0, engine='openpyxl')

        def remove_brands(x):
            for brand in self.brands:
                x = re.sub(fr'\b{brand}\b', '', x, flags=re.I)
            return x

        train_df['record_name'] = train_df['record_name'].apply(lambda x: remove_brands(x))
        return train_df

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
        loader = DataFrameLoader(self.ksr_df, page_content_column=self.page_content_column_bm25)
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

    def get_vocab_ksr(self):
        vocab_ksr = set([])
        for title in self.ksr_df['preproc_name'].values:
            vocab_ksr |= set(self.preproc_text(title).split(' '))
        for title in self.ksr_df['Наименование'].values:
            title = re.sub(r'\s+', ' ', title.lower())
            title = re.sub(r'[\.\,\'"=]', ' ', title)
            title = re.sub(r'[-)(\\\/]', ' ', title)
            vocab_ksr |= set(title.split(' '))
        vocab_ksr = vocab_ksr - set([''])
        return vocab_ksr

    def levenshteinRecursive(self, str1, str2, m, n):
        # str1 is empty
        if m == 0:
            return n
        # str2 is empty
        if n == 0:
            return m
        if str1[m - 1] == str2[n - 1]:
            return self.levenshteinRecursive(str1, str2, m - 1, n - 1)
        return 1 + min(
            # Insert
            self.levenshteinRecursive(str1, str2, m, n - 1),
            min(
                # Remove
                self.levenshteinRecursive(str1, str2, m - 1, n),
                # Replace
                self.levenshteinRecursive(str1, str2, m - 1, n - 1))
        )

    def jaccard_similarity(self, list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))

    def find_sim_word(self, word1):
        # Если слово есть в словаре синонимов, то используем его
        if word1.lower() in self.synonyms:
            return self.synonyms[word1.lower()]

        set_word = set(word1)
        sim_word = None
        sim_word_score = 0
        for vocab_word in self.vocab_ksr:
            jaccard_score = self.jaccard_similarity(set_word, set(vocab_word))
            # print(self.jaccard_similarity(set_word, set(vocab_word)))
            if jaccard_score > 0.7 and jaccard_score > sim_word_score:
                # levenshtein_distance = self.levenshteinRecursive(word1, vocab_word, len(word1), len(vocab_word))
                # if levenshtein_distance <= 1:
                #   return vocab_word
                sim_word_score = jaccard_score
                sim_word = vocab_word
        return sim_word

    # def is_number(self, s):
    #     try:
    #         float(s)
    #         return True
    #     except ValueError:
    #         return False
    def transliterate(self, word):
        # litters = { 'q': 'й', 'w': 'ц', 'e': 'у', 'r': 'к', 't': 'е', 'y': 'н', 'u': 'г', 'i': 'ш', 'o': 'щ', 'p': 'з', 'a': 'ф', 's': 'ы', 'd': 'в', 'f': 'а', 'g': 'п', 'h': 'р', 'j': 'о', 'k': 'л', 'l': 'д', 'z': 'я', 'x': 'ч', 'c': 'с', 'v': 'м', 'b': 'и', 'n': 'т', 'm': 'ь'}
        litters = {'q': 'й', 'w': 'ц', 'e': 'у', 'r': 'к', 't': 'е', 'y': 'н', 'u': 'г', 'i': 'ш', 'o': 'щ', 'p': 'з',
                   '[': 'х', ']': 'ъ', 'a': 'ф', 's': 'ы', 'd': 'в', 'f': 'а', 'g': 'п', 'h': 'р', 'j': 'о', 'k': 'л',
                   'l': 'д', ';': 'ж', "'": 'э', 'z': 'я', 'x': 'ч', 'c': 'с', 'v': 'м', 'b': 'и', 'n': 'т', 'm': 'ь',
                   ',': 'б', '.': 'ю'}
        word = word.lower()
        for key in litters:
            word = word.replace(key, litters[key])
        return word

    def replace_word(self, word):
        if word.lower() in self.vocab_ksr:
            return word.lower()
        # elif self.is_number(word):
        #    return word
        else:
            sim_word = self.find_sim_word(word)
            if sim_word:
                return sim_word.lower()
            else:
                transliterate_word = self.transliterate(word)
                if transliterate_word.lower() in self.vocab_ksr:
                    return transliterate_word.lower()
                else:
                    sim_word = self.find_sim_word(transliterate_word)
                    if sim_word:
                        return sim_word.lower()
                    else:
                        return ''

    #     def preporc_query_bm25(self):
    #     # train_df['record_name'].progress_apply(lambda x: ' '.join([word for word in split_text(x) if word in vocab_ksr ]))
    #     self.train_df['preproc_name_2'] = self.train_df['preproc_name'].progress_apply(lambda x:
    #                                 ' '.join(
    #                                         [self.replace_word(word) for word in self.preproc_text(x)]
    #                                 )
    #                             )

    def calc_conversion_factor(self, query, unit):
        # y во что нужно перевести
        if unit.lower() == 'т':
            match = re.findall(r'([\.\d]+)\s*?кг', query)
            if len(match) == 0:
                return 1
            conversion_factor = float(match[0]) / 1000
            return conversion_factor
        elif unit.lower() == 'шт':
            match = re.findall(r'(\d+)\s*?шт', query)
            if len(match) == 0:
                return 1
            conversion_factor = int(match[0])
            return conversion_factor
        else:
            return 1

    def preproc_titles(self):
        """
        Обрабатываем наименования
        """
        self.ksr_df['Наименование_original'] = self.ksr_df['Наименование']
        self.ksr_df['Наименование'] = self.ksr_df['Наименование'] + ' ' + self.ksr_df['Ед.изм.']
        self.ksr_df['Наименование'] = self.ksr_df['Наименование'].apply(lambda x: x.lower())
        self.train_df['record_name_lower'] = self.train_df['record_name'].apply(lambda x: x.lower())

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
        #         ksr_codes_llm, scores = self.predict_code_llm(query_text, k=self.k_len)
        #         ksr_codes_bm25 = self.predict_code_bm25(query_text)
        ksr_codes_llm, scores = self.predict_code_llm(query_text, k=self.k_len)

        query_text_bm25 = self.preproc_text(query_text)
        query_text_bm25 = [self.replace_word(word) for word in query_text_bm25.split(' ')]
        while '' in query_text_bm25:
            query_text_bm25.remove('')
        if len(query_text_bm25) < 2 or query_text_bm25 == ['']:
            return None, None
        query_text_bm25 = ' '.join(query_text_bm25)

        ksr_codes_bm25 = self.predict_code_bm25(query_text_bm25)

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
        query_text = query_text.lower()
        score_codes, count_codes = self.calc_score_codes(query_text)
        if score_codes == None:
            return ''
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
        query_text = query_text.lower()
        score_codes, _ = self.calc_score_codes(query_text)
        if score_codes == None:
            return 'Не найдено подходящее наименование в КСР, уточните запрос', '-', 0, 0

        for ksr_code in score_codes:
            final_score = score_codes[ksr_code]['place'] / score_codes[ksr_code]['score']
            score_codes[ksr_code]['final_score'] = final_score
        ksr_code, cos_sim, final_score = \
        pd.DataFrame.from_dict(score_codes, orient='index').sort_values(by='final_score').reset_index().iloc[0][
            ['index', 'score', 'final_score']].values
        select_ksr_df = self.ksr_df[self.ksr_df['Код ресурса'] == ksr_code]
        ksr_name = select_ksr_df['Наименование_original'].item()
        ksr_unit = select_ksr_df['Ед.изм.'].item()

        conversion_factor = self.calc_conversion_factor(query_text, ksr_unit)
        print(f'ksr_unit: {ksr_unit}')
        print(f'conversion_factor: {conversion_factor}')

        return ksr_code, ksr_name, cos_sim, final_score, conversion_factor

    #     def predict_code(self, query_text):


#         docs = self.bm25_retriever.get_relevant_documents(query_text)
#         return docs[0].metadata['Код ресурса']

if __name__ == '__main__':
    PATH = '.'
    DATASET_PATH = PATH + '/datasets'
    ksr_file = DATASET_PATH + '/classification.xlsx'
    train_file = DATASET_PATH + '/train.xlsx'

    stroy_model = StroyModel(
        ksr_file=ksr_file,
        train_file=train_file,
        # model=model,
        # embeddings=embeddings_KSR,
        model_name='intfloat/multilingual-e5-small',
        k_len=5,
        page_content_column_bm25='preproc_name',
        page_content_column_llm='Наименование'
    )

    query_text = 'Анкер забивной М10 DRM 12x40 сталь'
    print(stroy_model.predict_submit(query_text))
    print(stroy_model.predict_API(query_text))

