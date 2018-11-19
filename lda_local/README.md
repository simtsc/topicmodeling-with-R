# Topic modeling using (LDA) and parallelized cross-validation to determine number of topics.

Latent Dirichlet Allocation (LDA) is a popular model for topic modeling. Text data can become quite voluminous and needs to be transformed in various ways before it can be used to perform text analytics. This script uses the LDA implementation of the topicmodels library. In addition it is possible to configure the script to automatically determine the number of topics by itself, instead of using an explicit number of topics. The number of topics is chosen by minimizing average _perplexity_. All CPUs are used for parallel computation of different settings to improve runtime performance. Details are outlined below.

## Requirements

### Input data

The input file is expected to contain lines of text, where each line represents its own unit of independent text. Those lines are treated as one column with one text per row. Futher, the file is expected to have a header, i.e. name for the column. The input specification on the competition page took some liberty in classifying the input files as comma separated values (csv). In fact commas can appear anywhere as part of the textlines. The delimiter separating the lines is the `new-line` character.

### Libraries

The following R libraries are used by the script:
  - **optparse**: Create command line interface.
  - **tidyverse**: Collection of data science libraries.
  - **tidytext**: Text analysis tools.
  - **topicmodels**: Specialized library for topic modeling.
  - **SnowballC**: Word stemming.
  - **tm**: NLP tools.
  - **doParallel**: Parallelization library. 
  - **aws.s3**: Interfaces for AWS S3 buckets.

The script attampts to install the required libraries automatically, if they have not been installed before.

**Note**: The `topicmodels` library requires the GNU Scientific Library (GSL) to be installed on the system in order to compile all modules. The following command installs the GSL on CentOS:
```sh
sudo yum -y install gsl-devel
```

## Usage
```sh
Rscript --vanilla lda_local.R -h
```
The command above produces the following output.
```sh
Usage: lda_local.R [options]


Options:
        -f CHARACTER, --file=CHARACTER
                input csv file, may be a S3 bucket URL

        -n NUMBER, --numterms=NUMBER
                number of most relevant terms per topic [default = 3]

        -T NUMBER, --numtopics=NUMBER
                number of topics [default = 2]

        -k NUMBER, --cvfolds=NUMBER
                number of folds for cross-validation (CV) to estimate number of topics [default = -1]. Value must be > 2 for CV to be performed. Otherwise model is fit to numtopics

        -h, --help
                Show this help message and exit
```

### Examples

#### Run script with local input file

```sh
Rscript --vanilla lda_local.R -f input_data.csv
```

#### Run script with local input file and specifying number of terms and topics

```sh
Rscript --vanilla lda_local.R -f input_data.csv -T 3 -n 5
```

#### Run script with input file in S3

```sh
Rscript --vanilla lda_local.R -f s3://some-bucket/input_data.csv
```

#### Run script with local input file, determine number of topics by using cross-validation

```sh
Rscript --vanilla lda_local.R -f input_data.csv -k 5 -n 5
```

## Implementation

The implementation details and considerations are outlined in this section.

### Setup

The script parses the command line options after all libraries have been loaded. The number of terms and number of topics have to be equal or greater than 1. However, these arguments have sensible default values and do not need to be specified if the defaults suffice. The input file must be specified however. This argument is mandatory. Any violation of these conditions aborts the script and prints an error message.

### Read data

Data can be read from a local file as well as from a file located in a S3 bucket. The input file is copied from S3 to the local file system, if a S3 bucket URL has been given for the input file.

The first column of the resulting data frame is being renamed to `"text"` for consistency and all invalid lines, i.e. empty lines, are being removed.

In addition a dictionary is loaded which will be used later to complete word stems. The dictionaries have been obtained from open git repositories:

  - [dicts/intellij_english.dic](https://github.com/JetBrains/intellij-community/blob/master/spellchecker/src/com/intellij/spellchecker/english.dic)
  - [dicts/lstm_byronknoll_english.dic](https://github.com/byronknoll/lstm-compress/blob/master/dictionary/english.dic)

It is actually possible to derive a suitable dictionary from the input text itself. However, this has not been implemented in this script (yet).

### Cross-validation

As previously mentioned, --cvfolds or -k has to be set to an integer greater than 2, if cross-validation shall be performed. In this case the script ignores the explicit number of topics the user has set via --numtopics or -T.

The data is split into k folds, i.e. k number of subsets. The input data is sampled with replacement to create the folds. The algorithm uses a predefined set of topic numbers between 2 and 100. A heuristic could be used to control the search for an optimal number of topics. This has not been implemented (yet). _Perplexity_ is used as performance metric. The computation is parallelized over all available CPU cores. Each core computes the cross-validation score for one of the possible topic number values.

The last step is then to build a final model using all the input data and setting --numtopics to the value corresponding to the smallest average perplexity.

### Build and fit model

The data frame needs to be transformed to be usable by the LDA model. The quality of the topic model also depends on how the data has been filtered and transformed prior to fitting the model. The transformation stages in order of execution are:

1. Convert all strings to lower case.
2. Replace punctuation from all input strings with white space.
3. Remove all numbers from the input strings.
4. Remove stopwords from term list. Stopwords are taken from the SMART corpus.
5. Stem the remaining terms.
6. Keep only terms that have a minimum length of 3.
7. Construct a matrix with term counts.
8. Remove sparse terms, as they may not contribute well to finding common topics.

Finally the model is fit to the data.

Word stemming is done via the porter stemming algorithm. Details about the algorithm can be found on the [creators website](https://tartarus.org/martin/PorterStemmer/).

### Get topic and terms

Topics and their corresponding term frequencies (beta) are being retrieved from the model. The terms are being sorted by descending term frequency per topic.

### Create resulting data frame

The resulting data frame is being constructed by creating the topic name and their corresponding term frequencies as weights. The terms are being completed by using words from a dictionary since word stemming has been used earlier on. The resulting data frame has the following structure:

| topic | term | weight |
| - | - | - |
| Topic1 | foo | 0.12 |
| Topic1 | bar | 0.03 |
| Topic1 | baz | 0.01 |
| Topic2 | jaz | 0.4 |
| Topic2 | pop | 0.05 |
| Topic2 | rock | 0.022 |
