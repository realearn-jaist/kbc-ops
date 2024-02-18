@echo off
call conda activate env

python -u Baselines.py --embedding_type opa2vec --axiom_file axioms_hermit.txt --annotation_file annotations.txt --pretrained none --input_type concatenate

python -u Baselines.py --embedding_type rdf2vec --walker random --walk_depth 2

python -u Baselines.py --embedding_type onto2vec --axiom_file axioms_hermit.txt --pretrained none --input_type concatenate

python -u OWL2Vec_Plus.py --URI_Doc yes --Lit_Doc yes --Mix_Doc yes --Mix_Type random --Embed_Out_URI yes --Embed_Out_Words yes



