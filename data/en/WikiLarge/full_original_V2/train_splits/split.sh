
awk 'NR % 50000 == 1 {file = sprintf("wiki.full.aner.ori.train.detok.split_%d.src", int((NR-1)/50000) + 1)} {print > file}' /Users/sarubi/Desktop/A8/code/2/LLM_based_control_rewrite/data/en/WikiLarge/wiki.full.aner.ori.train.detok.src


awk 'NR % 50000 == 1 {file = sprintf("wiki.full.aner.ori.train.detok.split_%d.dst", int((NR-1)/50000) + 1)} {print > file}' /Users/sarubi/Desktop/A8/code/2/LLM_based_control_rewrite/data/en/WikiLarge/wiki.full.aner.ori.train.detok.dst
