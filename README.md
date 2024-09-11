# Sorting IMDB Top 250 Movies

# Business Problem¶
In this section, we are going to sort IMDB Top 250 movies while benefiting from
two different datasets and applying the following processes:
* Checking the datasets and getting general information
* MinMaxScaling the related features
* IMDB's ex-method for the average rating
* Bayesian average rating
* Combining the two datasets to create a new weighted average rating

# Required Libraries

    import math
    import warnings
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    from sklearn.preprocessing import MinMaxScaler
    
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 500)
    pd.set_option("display.float_format", lambda x: "%.2f" % x)
    
    warnings.filterwarnings('ignore')

# Importing the First Dataset

    movies_metadata = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Datasets/movies_metadata.csv")
    df = movies_metadata.copy()
    df.head()

## Updating the DataFrame

We update our dataframe as we don't need most of the columns. The columns that are necessary:

* title
* vote_average
* vote_count

      # New df
      df = df[["title", "vote_average", "vote_count"]]
      df.head()

|    | title                       |   vote_average |   vote_count |
|---:|:----------------------------|---------------:|-------------:|
|  0 | Toy Story                   |            7.7 |         5415 |
|  1 | Jumanji                     |            6.9 |         2413 |
|  2 | Grumpier Old Men            |            6.5 |           92 |
|  3 | Waiting to Exhale           |            6.1 |           34 |
|  4 | Father of the Bride Part II |            5.7 |          173 |

# Sorting the Movies by *vote_average*

The sorting is obviously faulty without taking **vote_count** into consideration.

      df.sort_values("vote_average", ascending=False).head()

|       | title                                           |   vote_average |   vote_count |
|------:|:------------------------------------------------|---------------:|-------------:|
| 21642 | Ice Age Columbus: Who Were the First Americans? |             10 |            1 |
| 15710 | If God Is Willing and da Creek Don't Rise       |             10 |            1 |
| 22396 | Meat the Truth                                  |             10 |            1 |
| 22395 | Marvin Hamlisch: What He Did For Love           |             10 |            1 |
| 35343 | Elaine Stritch: At Liberty                      |             10 |            1 |

As we can observe, most of the votes start at the 95% percent of the vote_count feature:

|              |   count |      mean |       std |   min |   10% |   20% |   30% |   40% |   50% |   60% |   70% |   80% |   90% |   *95%* |     99% |   max |
|:-------------|--------:|----------:|----------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|--------:|------:|
| vote_average |   45460 |   5.61821 |   1.92422 |     0 |   3.5 |   4.8 |   5.3 |   5.7 |     6 |   6.3 |   6.6 |     7 |   7.4 |   7.8 |    8.7  |    10 |
| vote_count   |   45460 | 109.897   | 491.31    |     0 |   1   |   2   |   4   |   6   |    10 |  15   |  25   |    50 | 160   | =>**434**   | 2183.82 | 14075 |

      df[df["vote_count"] > 434].sort_values("vote_average", ascending=False).head()

The sorting starts to make sense when the *vote_count* is considered. Popular and top movies start to be seen in the dataframe:

|       | title                       |   vote_average |   vote_count |
|------:|:----------------------------|---------------:|-------------:|
| 10309 | Dilwale Dulhania Le Jayenge |            9.1 |          661 |
| 40251 | Your Name.                  |            8.5 |         1030 |
|   314 | ***The Shawshank Redemption***    |            8.5 |         8358 |
|   834 | ***The Godfather***               |            8.5 |         6024 |
|  1176 | ***Psycho***                      |            8.3 |         2405 |

# Min-Max Scaling the Related Features

To comprehend and analyze the data better, we will apply **MinMaxScaler** on **vote_count** and **vote_average**.

      # Feature range is preferred the same as the IMDB rating range
      df["vote_count_score"] = MinMaxScaler(feature_range=(1,10)).fit(df[["vote_count"]]).transform(df[["vote_count"]])
      
      df.sort_values("vote_count_score", ascending=False).head(10)
      # Results are getting better

|       | title                   |   vote_average |   vote_count |   vote_count_score |
|------:|:------------------------|---------------:|-------------:|-------------------:|
| 15480 | Inception               |            8.1 |        14075 |           10       |
| 12481 | The Dark Knight         |            8.3 |        12269 |            8.84519 |
| 14551 | Avatar                  |            7.2 |        12114 |            8.74607 |
| 17818 | The Avengers            |            7.4 |        12000 |            8.67318 |
| 26564 | Deadpool                |            7.4 |        11444 |            8.31766 |
| 22879 | Interstellar            |            8.1 |        11187 |            8.15332 |
| 20051 | Django Unchained        |            7.8 |        10297 |            7.58423 |
| 23753 | Guardians of the Galaxy |            7.9 |        10014 |            7.40327 |
|  2843 | Fight Club              |            8.3 |         9678 |            7.18842 |
| 18244 | The Hunger Games        |            6.9 |         9634 |            7.16028 |

Now, we take **vote_average** into account.

      df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

      df.sort_values("average_count_score", ascending=False).head(10)

|       | title                   |   vote_average |   vote_count |   vote_count_score |   average_count_score |
|------:|:------------------------|---------------:|-------------:|-------------------:|----------------------:|
| 15480 | Inception               |            8.1 |        14075 |           10       |               81      |
| 12481 | The Dark Knight         |            8.3 |        12269 |            8.84519 |               73.415  |
| 22879 | Interstellar            |            8.1 |        11187 |            8.15332 |               66.0419 |
| 17818 | The Avengers            |            7.4 |        12000 |            8.67318 |               64.1815 |
| 14551 | Avatar                  |            7.2 |        12114 |            8.74607 |               62.9717 |
| 26564 | Deadpool                |            7.4 |        11444 |            8.31766 |               61.5507 |
|  2843 | Fight Club              |            8.3 |         9678 |            7.18842 |               59.6639 |
| 20051 | Django Unchained        |            7.8 |        10297 |            7.58423 |               59.157  |
| 23753 | Guardians of the Galaxy |            7.9 |        10014 |            7.40327 |               58.4858 |
|   292 | Pulp Fiction            |            8.3 |         8670 |            6.54387 |               54.3141 |

# IMDB Weighted Rating

This is the obsolete formula that IMDB has been using until 2015.

IMDB Ex Formula:

* v => vote_count
* M => required minimum vote_count
* r => vote_average
* C => constant value determined by IMDB

weighted_rating = (v/(v+M)* r) + (M/(v+M)* C)

Taking C into consideration as **7**

C = 7.0

example 1:

* r = 8
* M = 500
* v = 1000

weighted_rating = (1000/(1000+500)* 8) + (500/(1000+500)* 7) = 7.66

example 2:

* r = 8
* M = 500
* v = 3000

weighted_rating = (3000/(3000+500)* 8) + (500/(3000+500)* 7) = 7.86

      # IMDB formula as a function
      M = 2500
      C = df["vote_average"].mean()
      def imdb_weighted_rating(r, v, M, C):
        return (v/(v+M)*r) + (M/(v+M)*C)

      df["imdb_weighted_rating"] = imdb_weighted_rating(r=df["vote_average"], v=df["vote_count"], M=M, C=C)
      df.sort_values("imdb_weighted_rating", ascending=False).head(10)

|       | title                                             |   vote_average |   vote_count |   vote_count_score |   average_count_score |   imdb_weighted_rating |
|------:|:--------------------------------------------------|---------------:|-------------:|-------------------:|----------------------:|-----------------------:|
| 12481 | The Dark Knight                                   |            8.3 |        12269 |            8.84519 |               73.415  |                7.84604 |
|   314 | The Shawshank Redemption                          |            8.5 |         8358 |            6.34437 |               53.9271 |                7.83648 |
|  2843 | Fight Club                                        |            8.3 |         9678 |            7.18842 |               59.6639 |                7.74946 |
| 15480 | Inception                                         |            8.1 |        14075 |           10       |               81      |                7.72567 |
|   292 | Pulp Fiction                                      |            8.3 |         8670 |            6.54387 |               54.3141 |                7.69978 |
|   834 | The Godfather                                     |            8.5 |         6024 |            4.85194 |               41.2415 |                7.6548  |
| 22879 | Interstellar                                      |            8.1 |        11187 |            8.15332 |               66.0419 |                7.64669 |
|   351 | Forrest Gump                                      |            8.2 |         8147 |            6.20945 |               50.9175 |                7.59377 |
|  7000 | The Lord of the Rings: The Return of the King     |            8.1 |         8226 |            6.25996 |               50.7057 |                7.52155 |
|  4863 | The Lord of the Rings: The Fellowship of the Ring |            8   |         8892 |            6.68583 |               53.4866 |                7.47731 |

# Bayesian Average Rating (BAR)

This time, we import **another dataset** to benefit from BAR to sort the dataset.

Bayesian average rating formula:

![Alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/061e8b71312618ff414a60ab575232c9b6a878b4)

    # Bayesian average rating as a function
    def bayesian_average_rating(n, confidence=0.95):
        if sum(n) == 0:
            return 0
        K = len(n)
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        N = sum(n)
        first_part = 0.0
        second_part = 0.0
        for k, n_k in enumerate(n):
            first_part += (k + 1) * (n[k] + 1) / (N + K)
            second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
        score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
        return score

  # Importing the Second Dataset

    imdb_top = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Datasets/imdb_ratings.csv")
    df_2 = imdb_top.copy()
    df_2.head()

    #  We crop the first column because we don't need it in our DataFrame
    df_2 = df_2.iloc[:, 1:]
    df_2.head()

|    |     id | movieName                                |   rating |     ten |   nine |   eight |   seven |   six |   five |   four |   three |   two |   one |
|---:|-------:|:-----------------------------------------|---------:|--------:|-------:|--------:|--------:|------:|-------:|-------:|--------:|------:|------:|
|  0 | 111161 | 1.       The Shawshank Redemption (1994) |      9.2 | 1295382 | 600284 |  273091 |   87368 | 26184 |  13515 |   6561 |    4704 |  4355 | 34733 |
|  1 |  68646 | 2.       The Godfather (1972)            |      9.1 |  837932 | 402527 |  199440 |   78541 | 30016 |  16603 |   8419 |    6268 |  5879 | 37128 |
|  2 |  71562 | 3.       The Godfather: Part II (1974)   |      9   |  486356 | 324905 |  175507 |   70847 | 26349 |  12657 |   6210 |    4347 |  3892 | 20469 |
|  3 | 468569 | 4.       The Dark Knight (2008)          |      9   | 1034863 | 649123 |  354610 |  137748 | 49483 |  23237 |  11429 |    8082 |  7173 | 30345 |
|  4 |  50083 | 5.       12 Angry Men (1957)             |      8.9 |  246765 | 225437 |  133998 |   48341 | 15773 |   6278 |   2866 |    1723 |  1478 |  8318 |

Applying the BAR function:

    df_2["bar_score"] = df_2.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]]), axis=1)

    df_2.sort_values("bar_score", ascending=False).head(10)

Much more similar to the IMDB's current rating system:

|    |      id | movieName                                                     |   rating |     ten |   nine |   eight |   seven |   six |   five |   four |   three |   two |   one |   bar_score |
|---:|--------:|:--------------------------------------------------------------|---------:|--------:|-------:|--------:|--------:|------:|-------:|-------:|--------:|------:|------:|------------:|
|  0 |  111161 | 1.       The Shawshank Redemption (1994)                      |      9.2 | 1295382 | 600284 |  273091 |   87368 | 26184 |  13515 |   6561 |    4704 |  4355 | 34733 |     9.14539 |
|  1 |   68646 | 2.       The Godfather (1972)                                 |      9.1 |  837932 | 402527 |  199440 |   78541 | 30016 |  16603 |   8419 |    6268 |  5879 | 37128 |     8.94002 |
|  3 |  468569 | 4.       The Dark Knight (2008)                               |      9   | 1034863 | 649123 |  354610 |  137748 | 49483 |  23237 |  11429 |    8082 |  7173 | 30345 |     8.89596 |
|  2 |   71562 | 3.       The Godfather: Part II (1974)                        |      9   |  486356 | 324905 |  175507 |   70847 | 26349 |  12657 |   6210 |    4347 |  3892 | 20469 |     8.8125  |
|  4 |   50083 | 5.       12 Angry Men (1957)                                  |      8.9 |  246765 | 225437 |  133998 |   48341 | 15773 |   6278 |   2866 |    1723 |  1478 |  8318 |     8.76793 |
|  6 |  167260 | 7.       The Lord of the Rings: The Return of the King (2003) |      8.9 |  703093 | 433087 |  270113 |  117411 | 44760 |  21818 |  10873 |    7987 |  6554 | 28990 |     8.75204 |
|  5 |  108052 | 6.       Schindler's List (1993)                              |      8.9 |  453906 | 383584 |  220586 |   82367 | 27219 |  12922 |   6234 |    4572 |  4289 | 19328 |     8.74361 |
| 11 |  109830 | 12.       Forrest Gump (1994)                                 |      8.8 |  622104 | 553654 |  373644 |  151284 | 51140 |  22720 |  11692 |    7647 |  5941 | 12110 |     8.69915 |
| 12 | 1375666 | 13.       Inception (2010)                                    |      8.7 |  724798 | 627987 |  408686 |  174229 | 60668 |  26910 |  13436 |    8703 |  6932 | 17621 |     8.69315 |
| 10 |  137523 | 11.       Fight Club (1999)                                   |      8.8 |  637087 | 572654 |  371752 |  152295 | 53059 |  24755 |  12648 |    8606 |  6948 | 17435 |     8.67448 |

# Using both Ex-IMDB Formula and Bayesian Average Rating

We have to merge necessary features into one DataFrame to apply both formulas. But they differentiate from each other in some aspects.

Such as, **different titles**, **different score columns**.

Example of the difference:

|       | title (first dataframe)                                            | movieName (second dataframe)                                                    |
|------:|:--------------------------------------------------|:--------------------------------------------------------------|
| 12481 | The Dark Knight                                   | 4.       The Dark Knight (2008)                               |
|   314 | The Shawshank Redemption                          | 1.       The Shawshank Redemption (1994)                      |
|  2843 | Fight Club                                        | 11.       Fight Club (1999)                                   |
| 15480 | Inception                                         | 13.       Inception (2010)                                    |
|   834 | The Godfather                                     | 2.       The Godfather (1972)                                 |
|   351 | Forrest Gump                                      | 12.       Forrest Gump (1994)                                 |
|  7000 | The Lord of the Rings: The Return of the King     | 7.       The Lord of the Rings: The Return of the King (2003) |

    # updating the titles of the second dataframe
    new_title_list = []
    for title in df_2["movieName"]:
      new_title = title[4:-6].strip()
      new_title_list.append(new_title)
    df_2["movieName"] = new_title_list
    df_2.head()

Another problem is that the titles of the second dataframe contain their original title; on the other hand, the first dataframe has English versions.

For example:

| title (first dataframe)                                            | movieName (second dataframe)                                                    |
|:--------------------------------------------------|:--------------------------------------------------------------|
| A Beautiful Life                                   | La vita è bella                               |
| City of God                          | Cidade de Deus                      |
| Princess Mononoke                                        | Mononoke-hime                                  |


    # updating the first dataframe to have original titles
    df["original_title"] = movies_metadata["original_title"]
    df.sort_values("imdb_weighted_rating", ascending=False).head(10)

    # Adding imdb_weighted_rating to the second dataframe
    for title in df["original_title"]:
      if title in df_2["movieName"].values:
        df_2.loc[df_2["movieName"] == title, 'imdb_weighted_rating'] = df.loc[df["original_title"] == title, 'imdb_weighted_rating'].values[0]

    # Adding a new weighted average rating that contains both BAR and imdb formula
    df_2["total_weighted_rating"] = df_2['bar_score'] * 70/100 + df_2['imdb_weighted_rating'] * 30/100

|    |      id | movieName                                         |   rating |     ten |   nine |   eight |   seven |   six |   five |   four |   three |   two |   one |   bar_score |   imdb_weighted_rating |   total_weighted_rating |
|---:|--------:|:--------------------------------------------------|---------:|--------:|-------:|--------:|--------:|------:|-------:|-------:|--------:|------:|------:|------------:|-----------------------:|------------------------:|
|  0 |  111161 | The Shawshank Redemption                          |      9.2 | 1295382 | 600284 |  273091 |   87368 | 26184 |  13515 |   6561 |    4704 |  4355 | 34733 |     9.14539 |                7.83648 |                 8.75272 |
|  3 |  468569 | The Dark Knight                                   |      9   | 1034863 | 649123 |  354610 |  137748 | 49483 |  23237 |  11429 |    8082 |  7173 | 30345 |     8.89596 |                7.84604 |                 8.58099 |
|  1 |   68646 | The Godfather                                     |      9.1 |  837932 | 402527 |  199440 |   78541 | 30016 |  16603 |   8419 |    6268 |  5879 | 37128 |     8.94002 |                7.6548  |                 8.55445 |
| 12 | 1375666 | Inception                                         |      8.7 |  724798 | 627987 |  408686 |  174229 | 60668 |  26910 |  13436 |    8703 |  6932 | 17621 |     8.69315 |                7.72567 |                 8.40291 |
| 10 |  137523 | Fight Club                                        |      8.8 |  637087 | 572654 |  371752 |  152295 | 53059 |  24755 |  12648 |    8606 |  6948 | 17435 |     8.67448 |                7.74946 |                 8.39697 |
|  6 |  167260 | The Lord of the Rings: The Return of the King     |      8.9 |  703093 | 433087 |  270113 |  117411 | 44760 |  21818 |  10873 |    7987 |  6554 | 28990 |     8.75204 |                7.52155 |                 8.38289 |
|  7 |  110912 | Pulp Fiction                                      |      8.8 |  674884 | 541946 |  332876 |  140886 | 52091 |  26828 |  14203 |   10425 |  8912 | 25610 |     8.66717 |                7.69978 |                 8.37696 |
| 11 |  109830 | Forrest Gump                                      |      8.8 |  622104 | 553654 |  373644 |  151284 | 51140 |  22720 |  11692 |    7647 |  5941 | 12110 |     8.69915 |                7.59377 |                 8.36754 |
| 28 |  816692 | Interstellar                                      |      8.5 |  541682 | 412079 |  292240 |  149125 | 57253 |  24501 |  12271 |    7595 |  5618 | 12841 |     8.62289 |                7.64669 |                 8.33003 |
|  5 |  108052 | Schindler's List                                  |      8.9 |  453906 | 383584 |  220586 |   82367 | 27219 |  12922 |   6234 |    4572 |  4289 | 19328 |     8.74361 |                7.33338 |                 8.32054 |
|  2 |   71562 | The Godfather: Part II                            |      9   |  486356 | 324905 |  175507 |   70847 | 26349 |  12657 |   6210 |    4347 |  3892 | 20469 |     8.8125  |                7.1671  |                 8.31888 |
|  9 |  120737 | The Lord of the Rings: The Fellowship of the Ring |      8.8 |  631020 | 460809 |  316221 |  132929 | 46699 |  22658 |  11002 |    8295 |  6911 | 27013 |     8.66617 |                7.47731 |                 8.30951 |
| 13 |  167261 | The Lord of the Rings: The Two Towers             |      8.7 |  517453 | 420229 |  310443 |  130399 | 44743 |  20787 |   9690 |    6815 |  5851 | 20950 |     8.61475 |                7.41283 |                 8.25417 |
| 15 |  133093 | The Matrix                                        |      8.6 |  509155 | 503334 |  374602 |  165544 | 56990 |  25363 |  12552 |    8597 |  6496 | 16073 |     8.54647 |                7.40734 |                 8.20473 |
| 24 |   76759 | Star Wars                                         |      8.6 |  390842 | 329068 |  281958 |  133849 | 44539 |  18719 |   7973 |    5070 |  4063 | 16833 |     8.50988 |                7.43127 |                 8.1863  |
|  4 |   50083 | 12 Angry Men                                      |      8.9 |  246765 | 225437 |  133998 |   48341 | 15773 |   6278 |   2866 |    1723 |  1478 |  8318 |     8.76793 |                6.80594 |                 8.17934 |
| 19 |  114369 | Se7en                                             |      8.6 |  332914 | 489519 |  391070 |  157682 | 40010 |  14808 |   6786 |    4246 |  3124 |  7206 |     8.51991 |                7.36269 |                 8.17274 |
| 20 |  118799 | La vita è bella                                   |      8.6 |  198684 | 189402 |  133743 |   55912 | 19766 |   9232 |   4792 |    3011 |  2590 |  7419 |     8.57549 |                7.2086  |                 8.16542 |
| 26 |  120689 | The Green Mile                                    |      8.5 |  302739 | 358068 |  296957 |  122038 | 37322 |  14036 |   6627 |    3970 |  2581 |  5193 |     8.54252 |                7.23173 |                 8.14928 |
| 30 |  110413 | Léon                                              |      8.5 |  262783 | 322843 |  275578 |  115680 | 32902 |  11758 |   5145 |    2621 |  1918 |  5530 |     8.52568 |                7.24983 |                 8.14292 |









