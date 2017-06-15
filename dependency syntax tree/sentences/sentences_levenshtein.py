#coding:utf8


def levenshtein(aList, bList, ratio_flag=1):
    '''扩展的编辑距离算法，用以计算两个句子结构主干的相似度'''

    WEIGHT_1_LIST = ['HED', 'SBV', 'VOB', 'IOB', 'FOB', 'IS']   # 句子结构一级主干
    WEIGHT_2_LIST = ['LAD', 'RAD', 'ATT', 'ADV', 'COO']    # 句子结构二级主干
    WEIGHT_3_LIST = ['WP']     # 句子结构需要剔除的部分
    WEIGHT_1 = 3
    WEIGHT_2 = 2
    WEIGHT_3 = 1

    aList = [x for x in aList if x not in WEIGHT_3_LIST]
    bList = [x for x in bList if x not in WEIGHT_3_LIST]

    # 初始化数组a，根据主干等级来赋值
    lenA = len(aList)
    lenB = len(bList)
    a = [[0 for j in xrange(lenB+1)] for i in xrange(lenA+1)]

    temp_sum = 0
    for i in xrange(1, lenA+1):
        if aList[i-1] in WEIGHT_1_LIST:
            temp_sum += WEIGHT_1
        elif aList[i-1] in WEIGHT_2_LIST:
            temp_sum += WEIGHT_2
        else:
            temp_sum += WEIGHT_3
        a[i][0] = temp_sum

    temp_sum = 0
    for i in xrange(1, lenB+1):
        if bList[i-1] in WEIGHT_1_LIST:
            temp_sum += WEIGHT_1
        elif bList[i-1] in WEIGHT_2_LIST:
            temp_sum += WEIGHT_2
        else:
            temp_sum += WEIGHT_3
        a[0][i] = temp_sum

    # 填充数组a，采用动态规划算法
    for i in xrange(1, lenA+1):
        for j in xrange(1, lenB+1):

            # 1. 在aList上i位置删除字符（或者在bList上j-1位置插入字符）
            if aList[i-1] in WEIGHT_1_LIST:
                DEL_COST = WEIGHT_1
            elif aList[i-1] in WEIGHT_2_LIST:
                DEL_COST = WEIGHT_2
            else:
                DEL_COST = WEIGHT_3
            delete = a[i-1][j] + DEL_COST

            # 2. 在aList上i-1位置插入字符（或者在bList上j位置删除字符）
            if bList[j-1] in WEIGHT_1_LIST:
                INS_COST = WEIGHT_1
            elif bList[j-1] in WEIGHT_2_LIST:
                INS_COST = WEIGHT_2
            else:
                INS_COST = WEIGHT_3
            insert = a[i][j-1] + INS_COST

            # 3. 计算替换操作的代价，如果两个字符相同，则替换操作代价为0，否则为SUB_COST（根据最高句法结构等级来决定替换操作的代价）
            if aList[i-1] in WEIGHT_1_LIST or bList[j-1] in WEIGHT_1_LIST:
                SUB_COST = WEIGHT_1 * ratio_flag
            elif aList[i-1] in WEIGHT_2_LIST or bList[j-1] in WEIGHT_2_LIST:
                SUB_COST = WEIGHT_2 * ratio_flag
            else:
                SUB_COST = WEIGHT_3 * ratio_flag
            cost = aList[i-1] != bList[j-1] and SUB_COST or 0
            substitute = a[i-1][j-1] + cost

            a[i][j] = min(delete, insert, substitute)

    return (a[lenA][0] + a[0][lenB], a[lenA][lenB])


def levenshtein_distance(aList, bList):
    '''句子结构主干的编辑距离'''
    
    result = levenshtein(aList, bList)
    return result[1]


def levenshtein_ratio(aList, bList):
    '''句子结构主干的相似度'''

    result = levenshtein(aList, bList, 2)   # 替换操作代价乘以2
    ratio = (result[0] - result[1]) / (1.0 * result[0])
    return ratio


if __name__ == '__main__':
    start_time = time.time()
    #############

    a = ['HED', 'COO', 'LAD']
    b = ['HED', 'ADV', 'RAD','WP']
    print levenshtein_distance(a, b)
    print levenshtein_ratio(a, b)

    #############
    end_time = time.time()
    print 'Finished in %.2f secondes:' % (end_time - start_time)
