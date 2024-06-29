# KBCOps - Knowledge Base Completion Operations 

**Authors:** Teeradaj Racharak (JAIST, Japan) and Chavakan Yimmark (Mahidol University, Thailand) 

**Contacts:** racharak [at] jaist [dot] ac [dot] jp and chavakan [dot] yim [at] gmail [dot] com

**Disclaimer:** This repository contains executable source codes of our prototyped KBC Operations system. 

## Demo Video on Youtube

**Link:** https://youtu.be/gPTJiVcnG-I

## Online Application

**Link:** https://www.jaist.ac.jp/~racharak/lab/projects/kbc-ops

## Folder Documentation
This repository contains 2 codes, code used for experiment and KBCOps demo.
### Experiment code
The code used in our experiment are edited version of [OWL2Vec*](https://github.com/KRR-Oxford/OWL2Vec-Star) case study code.
These experiment codes can be found in [extraction](https://github.com/realearn-jaist/kbc-ops/tree/main/extraction) folder.
You can use the code to extract information into csv file for the KBCOps to analyse.
Noted that *infer* in csv file is what we called *garbage*.

### KBCOps
Excluding extraction code, these files are used to deploy KBCOps demo to our heroku sever using dash-framework as its base. You can find all the code used for the demo in [app.py](https://github.com/realearn-jaist/kbc-ops/blob/main/app.py).

## Paper 

**Title:** Are Embeddings All We Need for Knowledge Base Completion? Insights from Description Logicians

**Abstract:** Description Logic knowledge bases (KBs), i.e., ontologies, are often greatly incomplete, necessitating a demand for KB completion. Promising approaches to this aim are to embed KB elements such as classes, properties, and logical axioms into a low-dimensional vector space and and find missing elements by inferencing on the latent representation. Because these approaches make inference based solely on existing facts in KBs, the risk is that likelihood of KB completion with implicit (duplicated) facts could be high, making the performance of KB embedding models questionable. Thus, it is essential for the KB completion's procedure to prevent completing KBs by implicit facts. In this short paper, we present a new perspective of this problem based on the logical constructs in description logic. We also introduce a novel recipe for KB completion operations called **KBCOps** and include a demo that exhibits KB completion with fact duplication when using state-of-the-art KB embedding algorithms.


