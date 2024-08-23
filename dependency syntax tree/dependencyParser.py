import os,sys
from nltk.tokenize import StanfordTokenizer
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse import stanford
import  nltk
from IntelligentAssistantWriting.conf.params import *
from nltk.tag import StanfordPOSTagger
#linux
# stan_pos_jar_dir ='home/temp/stanfordPostaggerFull'

conf = create_params_lexile()
stan_pos_jar_dir = os.path.join(conf.dir_root,'temp/stanfordPostaggerFull')
poster_jar_path = os.path.join(stan_pos_jar_dir,'stanford-postagger.jar')
ebd_jar_path = os.path.join(stan_pos_jar_dir,'models/english-bidirectional-distsim.tagger')
eng_tagger = StanfordPOSTagger(model_filename=ebd_jar_path,path_to_jar=poster_jar_path)
# print(eng_tagger.tag('The quick brown fox jumped over the lazy dog'.split()))

#linux
# stan_jar_dir = '/home/enhui/temp/stanfordParserFull'
#windows
stan_jar_dir = os.path.join(conf.dir_root,'temp/stanfordParserFull')
parser_jar_path = os.path.join(stan_jar_dir,'stanford-parser.jar')
models_jar_path = os.path.join(stan_jar_dir,'stanford-parser-3.8.0-models.jar')
model_file = os.path.join(stan_jar_dir,'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

# eng_parser = StanfordDependencyParser(path_to_jar=parser_jar_path,path_to_models_jar=models_jar_path,model_path=model_file)
parser = stanford.StanfordParser(path_to_jar=parser_jar_path,path_to_models_jar=models_jar_path,model_path=model_file)
# sentences= parser.parse('The quick brown fox jumped over the lazy dog'.split())
# print sentences

def print_List(list_nums,sum,dep):
    sum = sum
    for each_item in list_nums :
        if isinstance(each_item,list):
            for item in each_item:
                if(isinstance(item,list)):
                       dep+=1
                       break
            sum = print_List(each_item,sum,dep)
        else:
            # print(each_item,dep)
            sum =dep+sum
    return  sum
if __name__=='__main__':
    sentences_1 = parser.raw_parse_sents(("The quick brown fox jumped over the lazy dog","Hello, My name is Melroy.", "What is your name?","We don't talk anymore, like we used to do."))
    sentences = parser.parse_one("The quick brown fox jumped over the lazy dog".split())
    content = "The Final Words of the Defense Lawyer Your Excellency,    In my conclusive announcement  , I'd like to appeal to you to make a verdict   that the boy is not guilty.  Following   is the reasonable doubt which shows the boy innocent.     Let's start with the evidence -- the switch knife, which is stabbed into the boy's father's chest.  According to Juror No. 8's proof, the strange switch knife can be bought in a little junk shop in the boy's neighbourhood.  It is not the only one which is sold in the neighbourhood store.  As to the downward angle of the stab wound, it is not strong enough to prove that the boy has done  it.  As we know, the boy is raised in a slum and he was even picked up for knife fighting, so he is experienced on   using knife  .  However, anyone who's ever used a switch knife will use it underhanded, but not stab downward. And secondly, there is something out of logic with the old man\'s testimony.  The old man said he had heard the boy scream out \"I\'m going to kill you\", and the sound of a body falling a second later.  But the EL gives out awful noises   when it goes by, how can the old man hear the words?  The old man\'s testimony also conflicts with the woman\'s testimony, on the point of the time the murder occuring    .  According to the woman\'s evidence, she saw the murder through the window of the EL\'s last two cars, that means the EL shall go by the old man\'s window for at least six seconds  before the body fell, but the old man said he heard the body fall a second after the boy\'s words -- if he really had heard the boy\'s words, he shall hear   it before the EL goes   by.  Moreover, there are some doubts on the boy\'s words \"I\'m going to kill you\", according to the old man\'s testimony.  Since the boy is not well-educated,  it is impossible for him to shout out a standard English sentence at the emotional time,if a person really wants to kill someone and has prepared a knife for it, he won\'t prefer to crying out his intention at the top of his lunge . Finally, let\'s get down to the woman\'s testimony.  The woman said she saw the murder while she was turning and tossing.  And she was sure what she had seen, because she had worn glasses.  But no people wear glasses to bed.  It is even impossible that the woman wore her glasses after she noticed something happening in the boy\'s father\'s room, for the last two cars of the EL passed her windows in two senconds   and after that   two senconds  , nothing can   be seen in the darkness.  So, if the woman did see something, she saw only a blur.   Sorry, I almost forget a detail about the male witness -- the detail seems so unimportant that perhaps you\'ve already forgottn   it as I do.  Meanwhile, it does provide us with useful information.  That is the old man said he ran to the door to see the boy running downstairs after fifteen seconds as he heard the body falling.  It is evident that the old   is crippled and confused half the time, how can he walk to the door with two canes   in such a short time and be positive about the exact time? Our junors   have already proved he can not act as he said in fifteen minutes.     The above is my final words on this case.  I appeal to  the judge to anounce   that the boy is not guilty.     Thank you."
    sentences = parser.raw_parse_sents(content)
    # text = 'The quick brown fox jumped over the lazy dog'
    # s_p = parser.parse_sents(text.split(),"Hello, My name is Melroy.".split())
    # print s_p
    # GUI
    dep = 0
    sum = 0
    print (print_List(sentences,sum,dep))
    # for line in sentences_1:
    #     print (line)
    # print ('ok')

