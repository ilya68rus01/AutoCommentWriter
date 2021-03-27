# from AutoCommentGetting import *
# from PreTrainingFiles import *
import numpy as np
import pandas as pd
import re
import nltk
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.text import Tokenizer

from AutoCommentGetting import AutoCommentGetting

bot = AutoCommentGetting()
bot.get_storys()
# data = bot.login_function(['https://pikabu.ru/story/vse_skhoditsya_dazhe_kianu_takoy_molodoy_7743043',
#                            'https://pikabu.ru/story/i_tak_vsegda_7756606',
#                            'https://pikabu.ru/story/uvolnenie_v_filmakh_7616411',
#                            'https://pikabu.ru/story/i_kinoroditelyami_veka_stanovyatsya_7744513',
#                            'https://pikabu.ru/story/stiven_spilberg_schitaet_russkiy_artkhaus_luchshim_filmom_7709672',
#                            'https://pikabu.ru/story/budto_v_film_popal_7463359',
#                            'https://pikabu.ru/story/yekspert_po_filmam_7664653',
#                            'https://pikabu.ru/story/khoroshie_filmyiantiutopii_kotoryie_provalilis_v_prokate_7733032',
#                            'https://pikabu.ru/story/alternativnaya_kontsovka_filma_krovavyiy_sport_7689467',
#                            'https://pikabu.ru/story/malenkie_khitrosti_rentv_7739343',
#                            'https://pikabu.ru/story/deystvitelno_a_kto_dolzhen_platit_za_okhranu_7835371',
#                            'https://pikabu.ru/story/raznitsa_vospriyatiya_7831773',
#                            'https://pikabu.ru/story/standartnaya_forma_7835975',
#                            'https://pikabu.ru/story/otvet_na_post_olive_7834931',
#                            'https://pikabu.ru/story/syin_risuet_nezhdannaya_radost_7831639',
#                            'https://pikabu.ru/story/ktoto_snyal_uskorennoe_24chasovoe_video_svoikh_rasteniy_chtobyi_pokazat_kak_silno_oni_lyubyat_dvigatsya_7834703',
#                            'https://pikabu.ru/story/sluchay_na_million_ili_kak_ya_kvartiru_snyal_7833824',
#                            'https://pikabu.ru/story/i_tak_v_90_sluchaev_7835963',
#                            'https://pikabu.ru/story/zakazala_lyustru_s_ali_yekspress_7837024',
#                            'https://pikabu.ru/story/afrodiziak_wd_40_ili_ne_dokumentirovannyiy_1002_sposob_primeneniya_7830663',
#                            'https://pikabu.ru/story/otvet_na_post_sluchay_na_million_7833256',
#                            'https://pikabu.ru/story/problemyi_s_obshcheniem_7835838',
#                            'https://pikabu.ru/story/badcomedian_obrashchenie_na_temu_sudov_shtrafov_i_tsenzuryi_6740346',
#                            'https://pikabu.ru/story/blackstar_i_pabla_ukrali_moyu_fotografiyu_i_ispolzuyut_v_kommercheskikh_tselyakh_6875084',
#                            'https://pikabu.ru/story/telekanal_rentv_uzhe_sovsem_obnaglel_5739743',
#                            'https://pikabu.ru/story/naskolko_gluboka_yeta_peshchera_7041398',
#                            'https://pikabu.ru/story/v_den_programmista_pro_logiku_pikabu_685289',
#                            'https://pikabu.ru/story/segodnya_611282',
#                            'https://pikabu.ru/story/kak_menya_izlechili_ot_isterichnoy_zhenyi_7301368',
#                            'https://pikabu.ru/story/moskovskogo_gaishnika_zastavili_pisat_obyasnitelnuyu__on_ostanovil_zamglavyi_politsii_moskvyi_obezzhavshego_probku_po_vstrechke_7184069',
#                            'https://pikabu.ru/story/styid_3022955',
#                            'https://pikabu.ru/story/true_story_3001448'])
#
#data.to_csv("data.csv", sep=';')

# test = PreTrainingFiles()
# test.convert_to_vec()
# test.predict_next_word("Сбился со счета сколько раз этот пост мне попадается в горячем за")
from tensorflow.python.keras.utils.np_utils import to_categorical
#
#
# class Test:
#     def __init__(self):
#         nltk.download('stopwords')
#         self.patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
#         self.stopwords_ru = stopwords.words("russian")
#         self.morph = MorphAnalyzer()
#         data_fr = pd.read_csv('dataframe.csv', delimiter=';')
#         data_fr2 = pd.read_csv('dataframe2.csv', delimiter=';')
#         data_fr3 = pd.read_csv('dataframe3.csv', delimiter=';')
#         data_fr4 = pd.read_csv('dataframe4.csv', delimiter=';')
#         data_fr = data_fr.merge(data_fr2, how='outer')
#         data_fr = data_fr.merge(data_fr3, how='outer')
#         self.data_frame = data_fr.merge(data_fr4, how='outer')
#         self.data_frame = pd.Series(data_fr['Comment'])[:1000]
#         print("awdawd")
#         self.data_frame = self.data_frame.dropna().drop_duplicates()
#         self.data_frame = self.data_frame.apply(self.__lemmatize__)
#         self.data_frame = self.data_frame.dropna()
#         (X, y) = self.split_sentence(self.data_frame)
#         maxWordsCount = 1000
#         tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
#                               lower=True, split=' ', char_level=False)
#         tokenizer.fit_on_sequences(X)
#         dist = list(tokenizer.word_counts.items())
#         print(dist[:10])
#         print("123")
#
#
#     def __lemmatize__(self, doc):
#         doc = re.sub(self.patterns, ' ', doc)
#         tokens = []
#         for token in doc.split():
#             if token and token not in self.stopwords_ru:
#                 token = token.strip()
#                 token = self.morph.normal_forms(token)[0]
#                 tokens.append(token)
#         if len(tokens) > 2:
#             return tokens
#         return None
#
#     def split_sentence(self, data):
#         x = list()
#         lst_x = list()
#         lst_y = list()
#         for words in data:
#             i = 0
#             z = 3
#             lst_x = list()
#             while i < len(words):
#                 if i == z:
#                     lst_y.append(words[i])
#                     i = i - 2
#                     z += 1
#                     x.append(lst_x)
#                     lst_x = list()
#                 lst_x.append(words[i])
#                 i += 1
#         return [x, lst_y]
#
# # test=Test()
# def split_sentence(data):
#     x = list()
#     lst_x = list()
#     lst_y = list()
#     for words in data:
#         i = 0
#         z = 3
#         lst_x = list()
#         while i < len(words):
#             if i == z:
#                 lst_y.append(words[i])
#                 i = i - 2
#                 z += 1
#                 x.append(lst_x)
#                 lst_x = list()
#             lst_x.append(words[i])
#             i += 1
#     return [x, lst_y]
#
# def split_sentence2(data):
#     x = np.zeros(shape=(2887392, 3))
#     lst_x = list()
#     arr_y = np.zeros(shape=(100, 25000))
#     j = 0
#     for i in range(np.size(data)):
#         z = 3
#         if i == z:
#             arr_y[j] = to_categorical(data[i], num_classes=25000)
#             i = i - 2
#             z+=1
#             x[j] = lst_x
#             j+=1
#             lst_x = list()
#         lst_x.append(data[i])
#     return [x, arr_y]
#
# with open('test.txt', 'r', encoding='utf-8') as f:
#     texts = f.read()
#     texts = texts.replace('\ufeff', '')
# maxWordsCount = 25000
# tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
#                           lower=True, split=' ', char_level=False)
# tokenizer.fit_on_texts([texts])
# dist = list(tokenizer.word_counts.items())
# print(dist[:10])
# dataqwe = tokenizer.texts_to_sequences([texts])
# print(dataqwe[0][:10])
# X = np.array([dataqwe[0][i:i+3] for i in range(2887392)])
# y = np.array([dataqwe[0][i] for i in range(3, 2887392)])
# y = to_categorical(y[:50000], num_classes=25000)
# x = np.zeros((2887390, 3))
# for i in range(2887390):
#     x[i] = X[i]
# print(x[:10])
# model2 = models.Sequential()
# model2.add(layers.Embedding(maxWordsCount, 128, input_length=3))
# model2.add(layers.SimpleRNN(128, return_sequences=True))
# model2.add(layers.SimpleRNN(64))
# model2.add(layers.Dense(500, activation='relu'))
# model2.add(layers.Dense(maxWordsCount, activation='softmax'))
# model2.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# model2.fit(x[:50000], y, epochs=1)
# (X, y) = split_sentence2(dataqwe[0])
# res = to_categorical(y[0], num_classes=maxWordsCount)
# qwer = np.zeros((100, 25000))
# for i in range(100):
#     qwer[i] = to_categorical(y[i], num_classes=maxWordsCount)
# model = models.Sequential()
# model.add(layers.Input(3))
# model.add(layers.Embedding(maxWordsCount, 256, input_length=3))
# model.add(layers.Dense(maxWordsCount, activation="softmax"))
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# print(model.summary())
# model.fit(X[:100], qwer, epochs=1)
#
