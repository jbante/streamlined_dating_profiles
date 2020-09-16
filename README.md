# Streamlined Dating Profiles
 
People looking for dating partners parse a burdensome volume of information, and much of that information fails to usefully distinguish one profile from the next. Dating profiles routinely include answers to hundreds, sometimes thousands, of questions about each person. Most of that information is necessarily redundant, in purely quantitative terms.

Rounding [the current world population](https://www.census.gov/popclock/) up to 7.7 billion, it should only take $\lceil log_2 7.7 \times 10^9 \rceil = 33$ well-chosen yes-or-no questions to uniquely identify every single living person on the entire planet. A lot makes each person who they are, but it doesn't take much to tell people apart. Dating profiles ought to require much less, considering that the dating pool for any particular person is much smaller than the global population.

Streamlining profile information might:
- Reduce information overload for audiences of dating profiles.
- Improve match quality by reducing distraction from meaningless information.
- Improve filtering efficiency.
- Reduce effort spent creating profiles by focusing on distinguishing information.

The challenge is identifying _which_ information is most useful for discriminating between prospective partners. This project tries 3 different approaches (in increasing order of description simplification):

- Select a subset of profile information as-given that is most effective at distinguishing between profiles (`2 - prioritized features.ipynb`).
- Synthesize raw features into a small number of profile meta-features that meaningfully summarize some trait (`3 - factor analysis.ipynb1).
- Cluster profiles into a reasonable number of "types" (`4 - types.ipynb`).

In the course of analyzing this data, I had to create new calculations for entropy and Hamming distance to handle the large volume of missing values in [the data set](https://doi.org/10.26775/ODP.2016.11.03) (`entropy with missing values.ipynb`), and I created a few variations on cross-validation suitable for unsupervised learning tasks (`unsupervised cross validation.ipynb`).