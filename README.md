<h3 align="center">Causality Detection</h3>

---

[Wissem Chabchoub](https://www.linkedin.com/in/wissem-chabchoub/) | [Contact us](mailto:chb.wissem@gmail.com)

## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About <a name = "about"></a>](#about)
- [🎥 Repository Structure  <a name = "repo-struct"></a>](#repo-struct)


## 🧐 About <a name = "about"></a>

In this project, we design and impelent a causality detection algorithm to find relationships between macroeconomic concepts (could be generalized to any concept). The work is based on [BERT(S) for Relation Extraction](https://github.com/plkmo/BERT-Relation-Extraction) in which the author performs a PyTorch implementation of the models for the paper "Matching the Blanks: Distributional Similarity for Relation Learning" published in ACL 2019. 

We alter this frameworl to mine only causal relations between two given entities and we add a framwork to automate the process of finding causal relations that could be used to build causal baysesian graphs. 

At first, you will need to fine tune the NLP model :

```
python train_inferer.py --train 1 --num_classes 3 --num_epochs 11
```

This requires SemEval2010 Task 8 dataset, available [here](https://github.com/sahitya0000/Relation-Classification/blob/master/corpus/SemEval2010_task8_all_data.zip). Download & unzip to ./inferer_src/data/ folder.

Then you can use the notebook to run the framwork :



## 🎥 Repository Structure  <a name = "repo-struct"></a>


1. `data`: Data forlder
2. `inferer_src `: Inferer source code
3. `src`: Main source code
4. `requirements.txt `: Requirements
5. `train_inferer.py`: To fine tune BERT
6. `reuters_corpus_causality.ipynb`: a Jupyter notebook for running and testing
