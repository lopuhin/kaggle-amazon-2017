Planet: Understanding the Amazon from Space
===========================================

Use satellite data to track the human footprint in the Amazon rainforest

This is my part of our team's solution for the Kaggle challange of
`Understanding the Amazon from Space <https://www.kaggle.com/c/planet-understanding-the-amazon-from-space>`_
Our team ods.ai `finished 7th <https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/leaderboard/private>`_.

We used common folds, and the data folder is assumed to be at ``../_data``,
so this is not runnable directly, but it should be easy to fix.

PyToch is used for all models, the most interesting bit is probably
the ``get_df_prediction`` function in ``make_bayes_submission.py``, which is an
alternative to adjusting thresholds and chooses the answer optimizing expected F2.
