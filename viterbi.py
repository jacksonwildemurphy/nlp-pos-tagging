# A program for efficiently finding the most likely part-of-speech tag sequence for a sentence.
# Based on the Viterbi algorithm.
# Requires 2 input files:
# 1) A text file of conditional probabilities for building the emission
#   and transition probability tables
# 2) A text file of sentences, 1 per line, to be POS tagged.
#
# Note: This program only supports the following tags: NOUN, VERB, INF, and PREP
#
# Written by Jackson Murphy. September 23, 2017

import sys
import numpy as np

# Takes a string representing a conditional probability "x y prob",
# converts the probability to a log probability base 2, and adds it to the table
def add_to_logprob_table(cond_prob, logprob_table):
    cond_prob_arr = cond_prob.split()
    x = cond_prob_arr[0]
    y = cond_prob_arr[1]
    logprob = np.log2(float(cond_prob_arr[2]))
    key = x + " " + y
    logprob_table[key] = logprob

# Returns a 2D matrix ("scores") of the best log probabilities for each word in a sentence,
# using the earlier generated conditional probability table.
# Also returns a 2D matrix ("back pointers") for recovering the sequences used to generate the "scores"
def find_seq_logprobs(sentence, tags):
    words = sentence.split()
    # Create the two 2D matrices
    w, h = len(words), len(tags);
    scores = [[0 for x in range(w)] for y in range(h)] # score probs are log2
    backptrs = [[0 for x in range(w)] for y in range(h)]
    # Initialize matrices
    for t in range(len(tags)):
        scores[t][0] = (get_cond_logprob(words[0],tags[t]) + get_cond_logprob(tags[t],"phi"))
        #print("first word probability of " + words[0] + " as " + tags[t] + ": " + str(scores[t][0]))
        backptrs[t][0] = 0
    #print("Scores: " + str(scores))
    # Fill out matrices
    for w in range(1, len(words)):
        for t in range(len(tags)):
            [max_score, max_index] = find_prev_word_max(scores, tags, w-1, tags[t])
            #print("max score of word before " + words[w] + " is: " + str(max_score) + " at index: " + str(max_index))
            scores[t][w] = get_cond_logprob(words[w],tags[t]) + max_score
            backptrs[t][w] = max_index
    return [scores, backptrs]


# Returns the requested conditional log2 probability stored in the table, i.e. (x | y),
# or log2(.0001) if the table contains no entry for that particular probability
def get_cond_logprob(x,y):
        key = x + " " + y
        if key not in logprob_table:
            return np.log2(.0001)
        else:
            return logprob_table[key]

# Gets the best POS tag for the previous word (max_idx), along with
# the previous word's log prob associated with that tag
def find_prev_word_max(scores, tags, prev_word_idx, tag):
    max_so_far = -(2 ** 63) - 1  # akin to minInt
    max_idx = -1
    for i in range(len(tags)):
        #print(str(scores[i][prev_word_idx]) + " " + str(get_cond_logprob(tag, tags[i])))
        prob = scores[i][prev_word_idx] + get_cond_logprob(tag, tags[i])
        #print("probability of find previous word max: " + str(prob))
        if prob > max_so_far:
            max_so_far = prob
            max_idx = i
    if max_idx == -1:
        print("Something went wrong in the find_prev_word_max function")

    return [max_so_far, max_idx]


def get_best_sequence(sentence, scores, backptrs, tags):
    # Find the best score for the last word in the sentence
    max_so_far = -(2 ** 63) - 1 # basically minInt
    max_idx = -1
    tag_count = len(scores)
    last_word_col = len(scores[0]) - 1
    for t in range(tag_count):
        logprob = scores[t][last_word_col]
        if logprob > max_so_far:
            max_so_far = logprob
            max_idx = t
    # Use back pointer matrix to extract sequence in reverse order
    # e.g. ["fish noun", "love verb", "I noun"]
    best_sequence = []
    words = sentence.split()
    t = max_idx  # t = tag index
    for w in range(len(scores[0])):
        best_sequence.append(words[len(words) - w - 1] + " " + tags[t])
        t = backptrs[t][w]
    best_sequence_logprob = max_so_far
    return [best_sequence, best_sequence_logprob]

def print_results(backptrs, best_sequence, best_sequence_logprob, scores, sentence, tags):
    print("PROCESSING SENTENCE: ", sentence)
    print("FINAL VITERBI NETWORK")
    words = sentence.split()
    for col in range(len(scores[0])):
        for row in range(len(scores)):
            print("P(" + words[col] + "=" + tags[row] + ") = " + "{0:.4f}".format(scores[row][col]))
    print("\nFINAL BACKPTR NETWORK")
    for col in range(len(scores[0])-1, 0, -1): # Iterate from last column to second column
        for row in range(len(scores)):
            print("Backptr(" + words[col] + "=" + tags[row] + ") = " + tags[backptrs[row][col]])
    print("\nBEST TAG SEQUENCE HAS LOG PROBABILITY = ", "{0:.4f}".format(best_sequence_logprob))
    for i in range(len(best_sequence)):
        word_tag = best_sequence[i].split()
        print(word_tag[0] + " -> " + word_tag[1])
    print()


###### START OF PROGRAM ######
if(len(sys.argv) < 3):
    print("This program requires 2 commandline arguments: <probabilities-file> <sentences-file> \n Please try again")
    sys.exit()

# Edit this list to handle more POS tags
tags = ["noun", "verb", "inf", "prep"]

# Build probabilities table from input file
logprob_table = {}  # combined transition and emission probability tables (in log base 2)
probabilities_file = open(sys.argv[1])
for line in probabilities_file:
    add_to_logprob_table(line.lower(), logprob_table)

# Read in sentences from input file, calculate their POS probabilities, id best sequence, print info
sentences_file = open(sys.argv[2])
for sentence in sentences_file:
    [scores, backptrs] = find_seq_logprobs(sentence.lower(), tags)
    [best_sequence, best_sequence_logprob] = get_best_sequence(sentence, scores, backptrs, tags)
    print_results(backptrs, best_sequence, best_sequence_logprob, scores, sentence, tags)
