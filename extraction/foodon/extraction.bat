@echo off
call conda activate env

python -u Baselines.py --embedding_type rdf2vec --walker wl --walk_depth 2 

python -u Baselines.py --embedding_type opa2vec --axiom_file axioms.txt --annotation_file annotations.txt --pretrained none

python -u Baselines.py --embedding_type onto2vec --axiom_file axioms.txt --pretrained none

python -u OWL2Vec_Plus.py --walker wl --walk_depth 4 --URI_Doc yes --Lit_Doc no --Embed_Out_URI yes --Embed_Out_Words no