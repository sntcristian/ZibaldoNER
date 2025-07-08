# ZibaldoNER

## Extract Silver Annotations for Named Entity Recognition from digitalzibaldone

This repository contains the source code used to extract silver annotations from the [DigitalZibaldone](https://digitalzibaldone.net/), a web platform that contains an XML/TEI editions of Giacomo Leopardi's *Zibaldone di pensieri*.

This project develops a simple Web Scraping strategy were entity annotations are extracted from the links in each 
paragraph. As an example, places referenced in the Zibaldone are linked to Wikidata. 

```html
<p> 
<a href="https://www.wikidata.org/wiki/Q220">Rome</a> is the capital of <a href="https://www.wikidata.org/wiki/Q38">Italy</a>.
</p>
```

By applying this algorithm, we automatically collected **2,941 references to persons, locations and bibliographic works** 
from 957 paragraphs ([available on Zenodo](https://doi.org/10.5281/zenodo.13971759)). 

## Named Entity Recognition Evaluation

Four Named Entity Recognition (NER) models were tested on the annotations of persons, locations and works from the aforementioned file:
* `LLaMa3.1-instruct-8B (Generative)`: Instruction-tuned LLM prompted in order to generate named entity annotations from text
* `LLaMa3.1-instruct-8B (Extractive)`: Instruction-tuned LLM prompted in order to extract a list of named entities from a text
* `GLiNER_ITA_BASE (Zero-shot)`: Pre-trained GliNER model trained on general-domain Italian for Universal Named Entity Recognition
* `GLiNER_ITA_BASE (Fine-tuned)`: Fine-tuned GliNER model trained on a [training portion](data/json_data/train.json) of Zibaldone
 

### Exact Matching Results

| Metric             | LLaMa Generative | LLaMa Extractive | GliNER Zero-Shot | GLiNER Fine-Tuned |
|--------------------|------------------------|------------------------|------------------------|-------------------------|
| Precision MicroAvg  | 22.48                  | 37.06                  | 30.60                  | **75.15**                   |
| Recall MicroAvg     | 48.42                  | 29.06                  | 50.79                  | **63.74**                   |
| F1 MicroAvg         | 30.71                  | 32.58                  | 38.19                  | **68.98**                   |


### Fuzzy Matching Results

| Metric             | LLaMa Generative  | LLaMa Extractive | GliNER Zero-Shot | GLiNER Fine-Tuned |
|--------------------|------------------------|------------------------|------------------------|-------------------------|
| Precision MicroAvg  | 24.73                  | 44.07                  | 35.33                  | **82.40**                   |
| Recall MicroAvg     | 53.27                  | 34.55                  | 58.64                  | **69.90**                   |
| F1 MicroAvg         | 33.78                  | 38.74                  | 44.09                  | **75.64**                   |



## Citation

```
@inproceedings{santini_named_2024,
	address = {Amsterdam, Netherlands},
	title = {Named {Entity} {Recognition} in {Historical} {Italian}: {The} {Case} of {Giacomo} {Leopardi}’s {Zibaldone}},
	volume = {3967},
	url = {https://ceur-ws.org/Vol-3967/X-TAIL-2024_paper_1.pdf},
	urldate = {2025-05-29},
	booktitle = {Posters and {Demos}, {Workshops}, and {Tutorials} of {EKAW} 2024},
	publisher = {CEUR},
	author = {Santini, Cristian and Melosi, Laura and Frontoni, Emanuele},
	year = {2024},
}
```



## References

* Santini, C. (2024). Zibaldoned: Silver annotations for Entity Disambiguation from Digitalzibaldone (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13971759

* Research - AI at Meta. (2024). The Llama 3 Herd of Models. Retrieved 6 September 2024, https://ai.meta.com/research/publications/the-llama-3-herd-of-models/

* Stoyanova, S. & Johnston, B. (Eds.), *Giacomo Leopardi's Zibaldone di pensieri: a digital research platform*. https://digitalzibaldone.net/

* Zaratiana, U., Tomeh, N., Holat, P., & Charnois, T. (2024). GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer. In K. Duh, H. Gomez, & S. Bethard (A c. Di), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) (pp. 5364–5376). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.naacl-long.300
