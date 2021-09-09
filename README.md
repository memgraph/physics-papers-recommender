# Physics paper recommender

This project is part of blog post on topic of `Recommendation System Using 
Online Node2Vec with Memgraph MAGE`. 

- An installation of [Memgraph Advanced Graph Extensions (MAGE)](https://memgraph.com/mage)
- An installation of [Memgraph Lab](https://memgraph.com/product/lab)  or 
usage of Memgraph's command-line tool, [mgconsole](https://docs.memgraph.com/memgraph/connect-to-memgraph/methods/mgconsole/), which is installed together with Memgraph.

# Setup
In order to prepare this repo, run:

```bash
pip3 install -r requirements.txt
```

# Prechecks
File `public/recommender.py` assumes existence of node2vec_online module, and calculated node embeddings.
If this is not prepared, follow blog post to learn how.

In order to check that you have `node2vec_online` query module loaded and embeddings ready, run
following command inside `Memgraph Lab` or `mgconsole`:
```cypher
CALL node2vec_online.get() YIELD *;
```

# Commands

Position yourself inside `public` repo

To visualize k-means inertia, run:
```bash
python3 recommender.py visualize
```

To get top 10 similarities over 5 groups, run:
```bash
python3 recommender.py similarities --top_n_sim=10 --n_clusters=5
```
