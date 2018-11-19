# Apache Spark, AWS and R: a scalable solution for training LDA models on large text corpora

Latent Dirichlet Allocation (LDA) is a popular model for topic modeling. Text data can become quite voluminous and needs to be transformed in various ways before it can be used to perform text analytics. This script offers a scalable solution for training a LDA model on a large corpus by utilizing _Apache Spark_ for distributed computation. The script can be run in **local** or **cluster** mode. The former installs Spark locally on the system that runs the script whereas the latter connects to a remote Spark cluster. Details are outlined below.

## Requirements

### Input data

The input file is expected to contain lines of text, where each line represents its own unit of independent text. In essence it is a table with one column and one text per row. Futher, the file is expected to have a header, i.e. a column name. The input specification on the competition page took some liberty in classifying the input files as comma separated values (csv). In fact commas can appear anywhere as part of the textlines. The delimiter separating the lines/rows is the `new-line` character.

### Libraries

The following R libraries are used by the script:
  - **optparse**: Create command line interface.
  - **dplyr**: Data transformation.
  - **sparklyr**: Interact with Apache Spark cluster.
  - **tm**: NLP tools.
  - **aws.s3**: Interfaces for AWS S3 buckets.

The script attampts to install all required libraries automatically, if they have not been installed before.

### Cluster

The connection is established via `livy` on port `8998` by default. Make sure, that the "Hadoop", "Spark" and "Livy" features are selected when creating the EMR instance.

If this is not an option the script can be run in local mode, i.e. the script will download and install a local spark instance. See example usage below for details. 

## Usage
```sh
Rscript --vanilla lda_spark.R -h
```
The command above produces the following output.
```sh
Usage: lda_spark.R [options]


Options:
        -f CHARACTER, --file=CHARACTER
                input csv file, may be a S3 bucket URL

        -s CHARACTER, --sparkhost=CHARACTER
                spark cluster address

        -m CHARACTER, --method=CHARACTER
                spark cluster connection method [default = livy]

        -v CHARACTER, --sparkversion=CHARACTER
                spark version [default = 2.3.0]

        -p NUMBER, --port=NUMBER
                port of spark cluster, depends on method [default = 8998]

        -n NUMBER, --numterms=NUMBER
                number of most relevant terms per topic [default = 3]

        -T NUMBER, --numtopics=NUMBER
                number of topics [default = 2]

        -h, --help
                Show this help message and exit
```

### Examples

#### Local mode

##### Run script with local input file and local mode

```sh
Rscript --vanilla lda_spark.R -f input_data.csv
```

**Note**: The script runs in local mode and downloads the desired spark version as indicated by `--sparkversion` automatically, since no cluster address has been provided. `--port` and `--method` will be ignored in this case.

##### Run script with local input file, local mode and specifying number of terms and topics

```sh
Rscript --vanilla lda_spark.R -f input_data.csv -T 3 -n 5
```

##### Run script with input file in S3 bucket and local mode

```sh
Rscript --vanilla lda_spark.R -f s3://some-bucket/input_data.csv
```

**Note**: The input file will be copied from the S3 bucket to the local system.

#### Cluster mode

Running the script in cluster mode works in the same way as the local mode examples above, with the addition of the adding the spark master address and optionally a different connection method and port.

##### Run script with input file in S3 and cluster mode
```sh
Rscript --vanilla lda_spark.R -f s3://some-bucket/input_data.csv -s master_url
```

**Note**: The connection is establied using `livy` on port `8998` by default. The cluster address will almost certainly vary.

## Implementation

The implementation details and considerations are outlined in this section.

### Setup

The script parses the command line options after all libraries have been loaded. The port number, number of terms and number of topics have to be equal or greater than 1. However, these arguments have sensible default values and do not need to be specified, if the defaults suffice. The input file must be specified. This argument is mandatory. Any violation of these conditions aborts the script and prints an error message.

### Connection to Spark cluster

The spark connection is established next. The connection mode depends on the provided arguments:

  - **local**: No spark cluster address was provided; the input file can be specified as local file or as s3 bucket URL.
  - **cluster**: A spark cluster address was provided; by default the connection attempt is made by using `livy` on port `8998`. The input file has to be copied to the Spark master manually, in case a local file path has been specified. Alternatively, a s3 bucket URL can be given in which case Spark retrieves the data directly from the designated bucket.

### Read data

Reading the data is handled by Spark. The input file is copied from S3 to the local system, if the script runs in local mode and a S3 bucket URL has been given for the input file.

The first column of the resulting data frame is being renamed to `"text"` for consistency and all invalid lines, i.e. empty lines, are being removed.

### Construct and run pipeline, fit model

The data frame needs to be transformed to be usable by the LDA model. The quality of the topic model also depends on how the data has been filtered and transformed prior to fitting the model. The transformation stages in order of execution are:

1. Replace punctuation with white space in all input strings.
2. Extract terms from strings which only contain letters and have a minimum length of 3. Strings are also converted to lower case in this stage.
3. Remove stopwords from term list. Stopwords are taken from the SMART corpus.
4. Construct a matrix with term counts. Remove sparse terms, as they may not contribute well to finding common topics.
5. Feed transformed data to LDA model estimator.

Finally the model is fit to the data.

### Get topic and terms

Topics and corresponding terms have to be retrieved from the model after successful fitting. `Stage 5` of the pipeline contains the fitted LDA model. Topic data resides on the cluster and is collected into local memory. Each topic is accompanied by a list of term indices and term weights. `Stage 4` contains the vocabulary, i.e. terms and their corresponding indices. This is needed to identify terms based on indicies returned by the model.

### Create resulting data frame

The resulting data frame is being constructed by creating the topic name and using the term indices to retrieve the corresponding terms and their weights. The resulting data frame has the following structure:

| topic | term | weight |
| ----- | ---- | ------ |
| Topic1 | foo | 0.12 |
| Topic1 | bar | 0.03 |
| Topic1 | baz | 0.01 |
| Topic2 | jaz | 0.4 |
| Topic2 | pop | 0.05 |
| Topic2 | rock | 0.022 |
