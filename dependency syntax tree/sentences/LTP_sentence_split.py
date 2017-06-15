#coding=utf8


def split_sentence_article(article):
    '''对文章进行句子切分，即至少包含两个paragraph。仅接受utf8编码'''

    result = []
    for paragraph in article.split('\n'):
        result += split_sentence_paragraph(paragraph)
    return result


def split_sentence_paragraph(paragraph):
    '''对段落进行句子切分。仅接受utf8编码'''

    paragraph = paragraph.decode('utf8')
    paragraph = ''.join(paragraph.split())
    words = paragraph

    result = [] # 句子列表
    line = []   # 当前句子
    i = 0
    while i < len(words):
        f = False   # 是否断句
        if i + 2 < len(words):
            if ((words[i] + words[i+1] + words[i+2] == u"？！”") or
                (words[i] + words[i+1] + words[i+2] == u"。’”") or
                (words[i] + words[i+1] + words[i+2] == u"……”") or   # 省略号，占两个字符
                (words[i] + words[i+1] + words[i+2] == u"……?") or   # 省略号，占两个字符
                (words[i] + words[i+1] + words[i+2] == u"……』") or   # 省略号，占两个字符
                (words[i] + words[i+1] + words[i+2] == u"！’”")):
                line.append(words[i]); i += 1
                line.append(words[i]); i += 1
                line.append(words[i]); i += 1
                result.append("".join([_.encode("utf-8") for _ in line]))
                line = []
                f = True

        if i + 1 < len(words):
            if ((words[i] + words[i+1] == u"。”") or
                (words[i] + words[i+1] == u"！”") or 
                (words[i] + words[i+1] == u"？”") or 
                (words[i] + words[i+1] == u"；”") or
                (words[i] + words[i+1] == u"……") or     # 省略号，占两个字符
                (words[i] + words[i+1] == u"。’") or
                (words[i] + words[i+1] == u"！’") or 
                (words[i] + words[i+1] == u"？’") or 
                (words[i] + words[i+1] == u"；’") or
                (words[i] + words[i+1] == u"？！") or
                (words[i] + words[i+1] == u"。』") or
                (words[i] + words[i+1] == u"！』") or 
                (words[i] + words[i+1] == u"？』") or 
                (words[i] + words[i+1] == u"；』")):
                line.append(words[i]); i += 1
                line.append(words[i]); i += 1
                result.append("".join([_.encode("utf-8") for _ in line]))
                line = []
                f = True

        if (not f and (words[i] == u"。" or
            words[i] == u"！" or 
            words[i] == u"？" or 
            words[i] == u"；")):
            line.append(words[i]); i += 1
            result.append("".join([_.encode("utf-8") for _ in line]))
            line = []
            f = True

        if not f:
            line.append(words[i])
            i += 1

    if len(line) > 0:
        result.append("".join([_.encode("utf-8") for _ in line]))

    # 输出句子列表
    # print type(sentence), sentence
    # for x in result:
        # print type(x), x
    return result


if __name__ == '__main__':
    sentence = '本以为秋天是收获的季节，原来秋天也是=,变了，变了，在这多……”彩的秋天里，一切都变了……生命的起点。\n每当秋天匆匆'
    split_sentence(sentence)
