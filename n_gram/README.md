## training ngram model

bin/lmplz -o 2 --verbose_header --skip_symbols --text corpus/kenlm_train_corpus_cut_sent_md.dat  --arpa mlmodel/lm.arpa_md_2  
bin/build_binary -s mlmodel/lm.arpa_md_2  mlmodel/lm.arpa_md_2.bin

## data format

氨 基 转 氨 酶  
膝 关 节 松 懈 术  

