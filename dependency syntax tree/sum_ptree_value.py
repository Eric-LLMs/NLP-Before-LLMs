import os,sys

def sorted_DictKey(adict):
    keys = adict.keys()
    keys.sort()
    return [dict[key] for key in keys]

def reverse_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=1)
    return [ backitems[i][1] for i in range(0,len(backitems))]

def sum_ptree_value(list_nums,sum,dep):
    sum = sum
    for each_item in list_nums :
        if isinstance(each_item,list):
            for item in each_item:
                if(isinstance(item,list)):
                       dep+=1
                       break
            sum = sum_ptree_value(each_item,sum,dep)
        else:
            # print(each_item),dep
            sum =dep+sum
    return  sum