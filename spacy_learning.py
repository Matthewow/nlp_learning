import spacy
from Logger import logging, get_logger
log = get_logger(__name__)


def sentencizer_test(nlp, txt):
    doc = nlp(txt)
    for sent in doc.sents:
        print(sent)


def tokenization_test(nlp, txt):
    doc = nlp(txt)
    tokens = [token for token in doc]
    print(tokens)

def pos_test(nlp, txt):
    doc = nlp(txt)
    for token in doc:
        print(token.text, token.pos_, token.tag_)


def lemmatization_test(nlp, txt):
    doc = nlp(txt)
    lem = [token.lemma_ for token in doc]
    print(doc,'\n', lem)


def stop_words_test(nlp, txt):
    log.info("Stop words are mostlt the most frequent words in a language, like 'be', 'that', and etc in English.")
    doc = nlp(txt)
    tokens = [token for token in doc]
    stop_words = [token.is_stop for token in doc]
    print(tokens)
    print(stop_words)

def dependency_parsing_test(nlp, txt):
    doc = nlp(txt)
    tokens = [token for token in doc]
    dep = [token.dep_ for token in doc]
    print(tokens)
    print(dep)


def noun_chunk_test(nlp, txt):
    doc = nlp(txt)
    noun_chunks = [nc for nc in doc.noun_chunks]
    print(noun_chunks)


def named_entity_recognization_test(nlp, txt):
    doc = nlp(txt)
    ners = [(ent.text, ent.label_) for ent in doc.ents]
    print(ners)


def word_vector_test(nlp, txt):
    doc = nlp(txt)
    print(doc[0].vector, type(doc[0].vector), doc[0].vector.shape)
    for token1 in doc:
        print(token1.has_vector, token1.vector_norm, token1.is_oov)
        for token2 in doc:
            print(token1.text, token2.text, token1.similarity(token2))



def run_all_testing():

    log.warning("========== sentencizer ===========")
    sentencizer_test(nlp_en, passage_en)
    sentencizer_test(nlp_zh, passage_zh)


    log.warning("========== tokenization ===========")
    tokenization_test(nlp_en, txt_en)
    tokenization_test(nlp_zh, txt_zh)

    log.warning("========== Part-of-speech tagging ========")
    pos_test(nlp_en, txt_en)
    pos_test(nlp_zh, txt_zh)

    log.warning("========== lemmatization_test ========")
    lemmatization_test(nlp_en, txt_en)
    lemmatization_test(nlp_zh, txt_zh)

    log.warning("========== stop_words_test ========")
    stop_words_test(nlp_en, txt_en)
    stop_words_test(nlp_zh, txt_zh)

    log.warning("========== dependency_parsing_test ========")
    dependency_parsing_test(nlp_en, txt_en)
    dependency_parsing_test(nlp_zh, txt_zh)

    log.warning("========== noun_chunk_test ========")
    noun_chunk_test(nlp_en, txt_en)
    # noun_chunk_test(nlp_zh, txt_zh)
    log.info("noun_chunk_test not support zh")

    log.warning("========== named_entity_recognization_test ========")
    named_entity_recognization_test(nlp_en, txt_en)
    named_entity_recognization_test(nlp_zh, txt_zh)

    log.warning("========== word_vector_test ========")
    log.info("en-------------")
    word_vector_test(nlp_en, txt_en)
    log.info("zh-------------")
    word_vector_test(nlp_zh, txt_zh)
        

def data_structures_test(nlp):
    # vocab
    log.info("vocab-----")
    coffee_hash = nlp.vocab.strings["coffee"] # type: int 
    coffee_string = nlp.vocab.strings[coffee_hash]   
    print(f"coffee_hash: {coffee_hash}, coffee_string: {coffee_string}")

    doc = nlp_en("I like coffee")
    lexeme = nlp_en.vocab["coffees"]
    print(lexeme.text, lexeme.orth, lexeme.is_alpha)
    
    #doc
    log.info("doc-----")
    from spacy.tokens import Doc, Span
    words = ["Hello", "world", "!"]
    spaces = [True, False, False]
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    span = Span(doc, 0, 2)
    span_with_label = Span(doc, 0, 2, label="GREETING")
    doc.ents = [span_with_label]
    print(doc.ents)


nlp_en = spacy.load('en_core_web_lg')
nlp_zh = spacy.load('zh_core_web_lg')

passage_en = "CuPy is an open-source array library accelerated with NVIDIA CUDA. CuPy provides GPU accelerated computing with Python. CuPy uses CUDA-related libraries including cuBLAS, cuDNN, cuRand, cuSolver, cuSPARSE, cuFFT and NCCL to make full use of the GPU architecture. The figure shows CuPy speedup over NumPy. Most operations perform well on a GPU using CuPy out of the box. CuPy speeds up some operations more than 100X. Read the original benchmark article Single-GPU CuPy Speedups on the RAPIDS AI Medium blog."
passage_zh = "新冠肺炎疫情在新年過後蠢蠢欲動，單日確診宗數昨（17日）反彈至雙位數，累計增至10,813宗，有多名患者恐拜年時播毒。今日（18日）年初七「人日」，料新增也有10宗左右。符合要求的食肆今晚起可恢復堂食至晚上10時，數以萬計飲食業員工連日來前往檢測，隱形患者隨之現形，兩間食肆廚房部有員工昨列確診，天后酒樓御名軒一名知客昨成為約10名初步確診者之一，不過政府未有撤回准許晚市堂食命令。"
txt_en = "Apple Pear He takes pride in turning his father’s excited ramblings about the latest civil rights incidents into handwritten business letters, and he was better in Google and Amazon."
txt_zh = "香港是国际金融、工商服务业及航运中心，美国传统基金会连续二十五年评选香港为全球最自由经济体"

# run_all_testing()
data_structures_test(nlp_en)

nlp = spacy.load("zh_core_web_sm")
doc = nlp("北京是一座美丽的城市")

