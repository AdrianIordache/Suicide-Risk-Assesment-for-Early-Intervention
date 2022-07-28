# Sucide Risk Assesment for Early Intervention
### BioMedical-NLP

Classifying Reddit Users in groups of suicide risk for possible early intervention based on their posts. 

Based on the previous work: ["Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention"](https://scholarcommons.sc.edu/cgi/viewcontent.cgi?article=1002&context=aii_fac_pub).

Reviewed here: [Paper Review](https://github.com/AdrianIordache/BioMedical-NLP/blob/master/papers/Original-Paper-Review.pdf).


Comparing Machine Learning aproches (e.g. Weighted Voting Ensambles) and Deep Learning models (e.g. Transformer Models: DistiBert).

The Machine Learning section was implemented by Adrian Iordache, and the Deep Learning side is based on the work of [Andrei Gîdea](https://github.com/andreiG98).

### Machine Learning Results

|                     | ***Accuracy*** | ***Precision*** | ***Recall*** | ***Ordinal Error*** |
|---------------------|-------------------|--------------------|-----------------|------------------------|
| LGBM                | 39.9\%            | 61.7\%             | 53.0\%          | 0.161                  |
| Random Forrest      | 43.1\%            | 64.8\%             | 56.3\%          | 0.128                  |
| AdaBoost            | 38.5\%            | 62.9\%             | 49.9\%          | 0.147                  |
| Logistic Regression | 43.1\%            | ***67.4\%***       | 54.5\%          | 0.161                  |
| SVC                 | 43.1\%            | 66.7\%             | 55\%            | 0.156                  |

### Weighted Voting Ensambles

|          | ***Accuracy*** | ***Precision*** | ***Recall*** | ***Ordinal Error*** |
|----------|-------------------|--------------------|-----------------|------------------------|
| SVC + LR | 44.7\%            | 54.6\%             | 71.2\%          | 0.048                  |
| RF + SVC | ***45.0\%***      | 53.7\%             | ***73.4\%***    | ***0.041***            |

### Deep Learning Results

| ***Architecture*** | ***Feature used***   | ***Accuracy*** | ***Precision*** | ***Recall*** | ***Ordinal Error*** |
|-------------------------------|---------------------|------------|---------------|-------------|------------------|
| DistilBert                    | Original Text           | 42.2\%            | 66.9\%             | 53.3\%          | 0.149                  |
| DistilBert                    | Social Prepocessed Text | 41.7\%            | 66.9\%             | 52.6\%          | 0.154                  |
