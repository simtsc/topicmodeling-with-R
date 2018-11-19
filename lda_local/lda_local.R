#######################################################################
## Script purpose: Create a LDA topic model using the topicmodels lib.
#######################################################################

#################### SETUP #######################
## Load required libraries, install if necessary.
##
## optparse:    Create command line interface.
## tidyverse:   Collection of data science libraries.
## tidytext:    Text analysis tools.
## topicmodels: Specialized library for topic modeling.
## SnowballC:   Word stemming.
## tm:          NLP tools.
## doParallel:  Parallelization library.
##################################################
for (package in c("optparse", "tidyverse", "tidytext", "topicmodels", "SnowballC", "tm", "doParallel")) {
  if (!require(package, character.only = T, warn.conflicts = F, quietly=T)) {
    install.packages(package, dependencies = T, repos = "http://cran.us.r-project.org")
    library(package, character.only = T, verbose = F, quietly = T)
    if (package == "tidytext") {
      devtools::install_github("juliasilge/tidytext", quiet = T)
    }
  }
}


arg_options <- list(
  make_option(c("-f", "--file"), type = "character", default = NULL, 
              help = "input csv file, may be a S3 bucket URL", metavar = "character"),
  make_option(c("-n", "--numterms"), type = "integer", default = 3, 
              help = "number of most relevant terms per topic [default = %default]", metavar = "number"),
  make_option(c("-T", "--numtopics"), type = "integer", default = 2, 
              help = "number of topics [default = %default]", metavar = "number"),
  make_option(c("-k", "--cvfolds"), type = "integer", default = -1, 
              help = paste("number of folds for cross-validation (CV) to estimate number of topics [default = %default].",
                           "Value must be > 2 for CV to be performed.",
                           "Otherwise model is fit to numtopics"), metavar = "number")
); 

opt_parser <- OptionParser(option_list=arg_options);
opt <- parse_args(opt_parser);

if (is.null(opt$file)){
  print_help(opt_parser)
  stop("An input file has not been specified", call. = F)
}

if (opt$numterms < 1 || opt$numtopics < 1){
  print_help(opt_parser)
  stop("Number of terms and topics must be greater than 0", call. = F)
}


################# READ DATA ######################
## Input data is expected to be a csv file.
##
## The data is copied from a S3 bucket to local
## file system, if a S3 bucket URL was provided.
## The first column of the resulting data frame
## is renamed to "text".
## Also a dictionary is being loaded for
## completing word stems.
##################################################
get_data <- function(f, delim = "\n", col_names = T, col_types = NULL) {
  input_file <- f
  if (startsWith(f, "s3://")) {
    if (!require("aws.s3", character.only = T, warn.conflicts = F, quietly = T)) {
      library("aws.s3", 
              lib.loc = "/usr/lib64/R/library", 
              character.only = T, 
              verbose = F, 
              quietly = T)
    }
    input_file <- save_object(f)
  }
  
  data <- read_delim(input_file, delim = delim, 
                     col_names = col_names,
                     col_types = col_types) %>%
    select(text = 1) %>%
    filter(!is.na(text)) %>% 
    as_tibble
  return(data)
}


text_data <- get_data(opt$file, col_types = "c")

dict_corpus <- get_data("dicts/intellij_english.dic", 
                        col_names = "term", 
                        col_types = "c") %>% 
  VectorSource %>% 
  VCorpus


############## BUILD AND FIT MODEL ##############
## The input data is being transformed and fed to
## the LDA model. Use cross-validation with 
## multiple folds to find the best number
## of topics instead of taking a fixed, 
## given one.
## If cross-validation is used, then perplexity
## is used as performance metric.
##
## Transformation stages:
## 1. Convert all strings to lower case.
## 2. Replace punctuation from all input strings 
##    with white space.
## 3. Remove all numbers from the input strings.
## 4. Remove stopwords from term list. Stopwords 
##    are taken from the SMART corpus.
## 5. Stem the remaining terms.
## 6. Keep only terms that have a minimum 
##    length of 3.
## 7. Construct a matrix with term counts.
## 8. Remove sparse terms, as they may not 
##    contribute well to finding common topics.
##
## Finally the model is fit to the data.
##################################################
control_LDA_Gibbs <- list(alpha = 50/opt$numtopics + 1, 
                          estimate.beta = T, 
                          verbose = 0, 
                          prefix = tempfile(), 
                          save = 0, 
                          keep = 50, 
                          seed = 42, 
                          nstart = 1, 
                          best = T, 
                          delta = 0.1, 
                          iter = 2000, 
                          burnin = 100, 
                          thin = 2000)

control_dtm <- list(tolower = T,
                    language = "english",
                    removePunctuation = T,
                    removeNumbers = T,
                    stopwords = stopwords(kind = "SMART"),
                    stemming = T,
                    wordLengths = c(3, Inf))

folds <- opt$cvfolds
k_min <- opt$numtopics
sparsity <- 0.95

# check, if cross-validation is to be used.
if (folds > 2) {
  # create CPU cluster, use all available cores
  cluster <- makeCluster(detectCores(logical = T))

  registerDoParallel(cluster)

  # make sure all required R libraries are available in each process
  clusterEvalQ(cluster, {
    library("dplyr", character.only = T, verbose = F, quietly = T)
    library("tm", character.only = T, verbose = F, quietly = T)
    library("topicmodels", character.only = T, verbose = F, quietly = T)
    library("tidytext", character.only = T, verbose = F, quietly = T)
  })

  # create data indicies to consider for each fold
  splitfolds <- sample(1:folds, nrow(text_data), replace = TRUE)
  # test these number of topics, a better heuristic could be used
  candidate_k <- c(2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100)
  # send variables to CPUs as well
  clusterExport(cluster, c("text_data", "control_LDA_Gibbs", "control_dtm", 
                           "splitfolds", "folds", "candidate_k", "sparsity"))

  results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar% {
    k <- candidate_k[j]
    
    control_LDA_Gibbs['alpha'] <- 50/k + 1
  
    results_1k <- data.frame(k = numeric(folds), perplexity = numeric(folds))
    
    for(i in 1:folds) {
      train_set <- text_data[splitfolds != i, ]
      valid_set <- text_data[splitfolds == i, ]

      dtm_train <- VCorpus(VectorSource(train_set)) %>% 
        DocumentTermMatrix(control = control_dtm) %>%
        removeSparseTerms(sparsity)
      dtm_valid <- VCorpus(VectorSource(valid_set)) %>%
        DocumentTermMatrix(control = control_dtm) %>%
        removeSparseTerms(sparsity)
    
      fitted <- LDA(dtm_train, k = k, method = "Gibbs", control = control_LDA_Gibbs)
      results_1k[i,] <- list(k, perplexity(fitted, newdata = dtm_valid))
    }
    return(results_1k)
  }
  stopCluster(cluster)

  results_df <- as.data.frame(results)
  # calculate the mean for each number of topics
  avg_df <- aggregate(results_df[, 2], list(results_df$k), mean)
  k_min <- avg_df[which.min(avg_df$x),]$Group.1
  control_LDA_Gibbs['alpha'] <- 50/k_min + 1
}


################### FINAL MODEL ##################
## Use chosen parameters and all data to fit
## final model.
##################################################
lda <- VCorpus(VectorSource(text_data)) %>%
  DocumentTermMatrix(control = control_dtm) %>% 
  removeSparseTerms(sparsity) %>%
  LDA(k = k_min, method = "Gibbs", control = control_LDA_Gibbs)


############## GET TOPICS AND TERMS ##############
## Extract topics and terms from the fitted model.
##
## topics: Topics and their corresponding term 
##         frequencies (beta) are being retrieved
##         from the model.
## terms:  The terms are being sorted by 
##         descending term frequency per topic.
##################################################
topics <- tidy(lda, matrix = "beta")

top_terms <- topics %>%
  group_by(topic) %>%
  top_n(opt$numterms, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)


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
res_num <- k_min * opt$numterms
res_df <- data.frame(topic = character(res_num), 
                     term = character(res_num), 
                     weight = numeric(res_num), 
                     stringsAsFactors = F)
for (row in 1:nrow(top_terms)) {
  topic <- paste0("Topic", top_terms[row, "topic"] %>% pull)
  weight <- top_terms[row, "beta"] %>% pull
  term <- top_terms[row, "term"] %>% pull
  term_completed <- stemCompletion(term, 
                                   dictionary = dict_corpus, 
                                   type = "prevalent")
  
  if (is.na(term_completed)) {
    term_completed <- term
  }
  
  res_df[row,] <- list(topic, term_completed, weight)
}

print(res_df)
