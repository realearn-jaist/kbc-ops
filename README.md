# KBCOps - Knowledge Base Completion Operations 

**Authors:** Teeradaj Racharak (JAIST, Japan) and Chavakan Yimmark (Mahidol University, Thailand) 

**Contacts:** racharak [at] jaist [dot] ac [dot] jp and chavakan [dot] yim [at] gmail [dot] com

**Disclaimer:** This repository contains executable source codes supporting our submission to IJCAI 2024 (Demo Track). 

## Demo Video on Youtube

**Link:** https://youtu.be/WWNdhqg19hs

## Online Application

**Link:** https://www.jaist.ac.jp/~racharak/lab/projects/kbc-ops

## Folder Documentation
TBD 

## Paper 

**Title:** Are Embeddings All We Need for Knowledge Base Completion? Insights from Description Logicians

**Abstract:** Description Logic knowledge bases (KBs), i.e., ontologies, are often greatly incomplete, necessitating a demand for KB completion. Promising approaches to this aim are to embed KB elements such as classes, properties, and logical axioms into a low-dimensional vector space and and find missing elements by inferencing on the latent representation. Because these approaches make inference based solely on existing facts in KBs, the risk is that likelihood of KB completion with implicit (duplicated) facts could be high, making the performance of KB embedding models questionable. Thus, it is essential for the KB completion's procedure to prevent completing KBs by implicit facts. In this short paper, we present a new perspective of this problem based on the logical constructs in description logic. We also introduce a novel recipe for KB completion operations called **KBCOps** and include a demo that exhibits KB completion with fact duplication when using state-of-the-art KB embedding algorithms.
