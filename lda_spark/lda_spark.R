#######################################################################
## Script purpose: Create a LDA topic model using Apache Spark.
#######################################################################

#################### SETUP #######################
## Load required libraries, install if necessary.
##
## optparse: Create command line interface.
## dplyr:    Data transformation.
## sparklyr: Interact with Apache Spark cluster.
## tm:       NLP tools.
##################################################
for (package in c("optparse", "dplyr", "sparklyr", "tm")) {
  if (!require(package, character.only = T, warn.conflicts = F, quietly = T)) {
    install.packages(package, dependencies = T, repos = "http://cran.us.r-project.org")
    library(package, character.only = T, verbose = F, quietly = T)
  }
}

arg_options <- list(
  make_option(c("-f", "--file"), type = "character", default = NULL, 
              help = "input csv file, may be a S3 bucket URL", metavar = "character"),
  make_option(c("-s", "--sparkhost"), type = "character", default = NULL, 
              help = "spark cluster address", metavar = "character"),
  make_option(c("-m", "--method"), type = "character", default = "livy", 
              help = "spark cluster connection method [default = %default]", metavar = "character"),
  make_option(c("-v", "--sparkversion"), type = "character", default = "2.3.0", 
              help = "spark version [default = %default]", metavar = "character"),
  make_option(c("-p", "--port"), type = "integer", default = 8998, 
              help = "port of spark cluster, depends on method [default = %default]", metavar = "number"),
  make_option(c("-n", "--numterms"), type = "integer", default = 3, 
              help = "number of most relevant terms per topic [default = %default]", metavar = "number"),
  make_option(c("-T", "--numtopics"), type = "integer", default = 2, 
              help = "number of topics [default = %default]", metavar = "number")
); 

opt_parser <- OptionParser(option_list = arg_options);
opt <- parse_args(opt_parser);

if (is.null(opt$file)){
  print_help(opt_parser)
  stop("An input file has not been specified", call. = F)
}

if (opt$numterms < 1 || opt$numtopics < 1){
  print_help(opt_parser)
  stop("Number of terms and topics must be greater than 0", call. = F)
}

if (opt$port < 1){
  print_help(opt_parser)
  stop("Invalid port number", call. = F)
}


########### CONNECT TO SPARK CLUSTER #############
## Establish connection to local or remote spark
## cluster.
##
## local: Spark will be installed and run locally.
##        This is done automatically by sparklyr.
## remote: A Spark cluster.
##################################################
get_spark_connection <- function(host = NULL, port = NULL, method = NULL, version = NULL) {
  if (is.null(host)) {
    spark_install(version)
    sc <- spark_connect(master = "local")
    return(sc)
  }
  else {
    spark_con_string <- paste0(host, ":", port)
    sc <- spark_connect(
      master = spark_con_string,
      method = method)
    return(sc)
  }
}

sc <- get_spark_connection(opt$sparkhost, opt$port, opt$method, opt$sparkversion)

if (!spark_connection_is_open(sc)) {
  stop(paste("Could not connect to spark cluster at", spark_con_string, "via", opt$method), call. = F)
}


################# READ DATA ######################
## Input data is expected to be a csv file.
##
## The input file is copied to the local system,
## if no spark host address has been specified
## (local mode) and the file resides in an AWS S3
## bucket. A local file path can be used, if the
## file is already located on the Spark master.
## The first column is renamed to "text" and rows
## with NA are removed.
##################################################
input_file <- opt$file

if (is.null(opt$sparkhost) && startsWith(opt$file, "s3://")) {
  if (!require("aws.s3", character.only = T, warn.conflicts = F, quietly = T)) {
    library("aws.s3", 
            lib.loc = "/usr/lib64/R/library", 
            character.only = T, 
            verbose = F, 
            quietly = T)
  }
  input_file <- save_object(opt$file)
}

text_data <- spark_read_csv(sc, 
                            "hackathon_topicmodeling", 
                            input_file,
                            header = T,
                            delimiter = "\n")

text_data <- text_data %>% select(text = 1) %>% filter(!is.na(text))


##### CONSTRUCT AND RUN PIPELINE, FIT MODEL #####
## The input data is being transformed and fed to
## the LDA model.
##
## Transformation pipeline stages:
## 1. Replace punction from all input strings 
##    with white space.
## 2. Extract terms from strings which only
##    contain letters and have a minimum
##    length of 3. Strings are also converted to
##    lower case in this stage.
## 3. Remove stopwords from the term lists.
##    Stopwords are taken from the SMART corpus.
## 4. Construct a matrix with term counts. Remove 
##    sparse terms, as they may not contribute 
##    well to finding common topics.
## 5. Feed transformed data to LDA model estimator.
##
## Finally the model is fit to the data.
##################################################
remove_punctuation <- text_data %>% mutate(text = regexp_replace(text, "[_\"\'():;,.!?\\-]", " "))

pipeline <- ml_pipeline(sc) %>%
  ft_dplyr_transformer(remove_punctuation) %>%
  ft_regex_tokenizer(input_col = "text", output_col = "word_list", pattern="[A-Za-z]+", gaps = F, min_token_length = 3) %>%
  ft_stop_words_remover(input_col = "word_list", output_col = "word_list_cleaned", stopwords = stopwords(kind = "SMART")) %>%
  ft_count_vectorizer(min_tf = 1.0, input_col = "word_list_cleaned", output_col = "features") %>%
  ml_lda(k = opt$numtopics, seed = 42, subsampling_rate = 0.0125, max_iter = 80, optimize_doc_concentration = T)

pipeline_model <- pipeline %>% ml_fit(text_data)


############## GET TOPICS AND TERMS ##############
## Extract topics and terms from the fitted model.
##
## topics: Stage 5 of the pipeline contains the 
##         fitted LDA model. Topic data resides on
##         the cluster and is collected into local
##         memory.
## terms:  Stage 4 contains the vocabulary, i.e. 
##         terms and their corresponding indices. 
##         This is needed to identify terms based
##         on indices returned by the model.
##################################################
topics <- ml_describe_topics(ml_stage(pipeline_model, 5), max_terms_per_topic = opt$numterms) %>% 
  mutate(topic = topic + 1) %>% 
  collect()
terms <- ml_stage(pipeline_model, 4)$vocabulary


############ DISCONNECT FROM CLUSTER #############
## The Spark cluster is not needed from this point
## onwards, as all results reside in local memory.
##################################################
spark_disconnect(sc)


########## CREATE RESULTING DATA FRAME ###########
## Build resulting dataframe.
##
## example
## topic  |  term  |  weight
## Topic1 |   foo  |    0.12
## Topic1 |   bar  |    0.03
## Topic1 |   baz  |    0.01
## Topic2 |   jaz  |    0.4
## Topic2 |   pop  |    0.05
## Topic2 |   rock |    0.022
##################################################
res_num <- opt$numtopics * opt$numterms
res_idx <- 0
res_df <- data.frame(topic = character(res_num), 
                     term = character(res_num), 
                     weight = numeric(res_num), 
                     stringsAsFactors = F)
for (row in 1:nrow(topics)) {
  topic <- paste0("Topic", topics[row, "topic"] %>% pull)
  termIdxs <- unlist(topics[row, "termIndices"]) + 1
  termWeights <- unlist(topics[row, "termWeights"])
  
  tterms <- terms[termIdxs]
  
  for (i in 1:length(termWeights)) {
    res_idx <- res_idx + 1
    res_df[res_idx,] <- list(topic, tterms[i], termWeights[i])
  }
}

print(res_df)
