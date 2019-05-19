#!usr/bin/python
# (c) 2013 Ryan Boehning | edited 2019 by Nico Schmdit


'''A Python implementation of the Smith-Waterman algorithm for local alignment
of nucleotide sequences.
Script copied and adapted from https://gist.github.com/radaniba/11019717
'''

import heapq
import numpy as np

# These scores are taken from Wikipedia.
# en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
# we take them as default settings
# match    = 2
# mismatch = -1
# gap      = -1


identity = lambda s:s

class SmithWaterman(object):
    
    def __init__(self,
                 match_score=2,
                 mismatch_score=-1,
                 gap_score=-1,
                 n_max_alignments=5,
                 min_score_treshold=10,
                 string_mapping_function=identity):
        
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_score = gap_score
        self.n_max_alignments = n_max_alignments
        self.min_score_treshold = min_score_treshold
        self.string_mapping_function = string_mapping_function
        
        self.score_matrix = np.empty((0,0))
        self.max_score = []
        self.visited_pos = []
        
        
    def align(self, seq1, seq2):
            
        # Initialize the scoring matrix.
        self.create_score_matrix(seq1, seq2)
        
        n_positive_scores = len(self.max_score)
        n_matches = 0
        k = 0
        alignments = []
        while n_matches < self.n_max_alignments and k < n_positive_scores: 
            k+=1
            (score, start_pos) = heapq.heappop(self.max_score)
            score = -score # note: score was saved as negative for descending order of priority queue
            if score < self.min_score_treshold: # consider not successful matches
                break
            
            # Traceback. Find the optimal path through the scoring matrix. This path
            # corresponds to the optimal local sequence alignment.
            alignment, seq1_aligned, seq2_aligned, valid = self.traceback(start_pos, seq1, seq2)
            if not valid:
                continue
            
            n_matches += 1
            alignments.append((score, alignment, seq1_aligned, seq2_aligned))
        
        return alignments




    def create_score_matrix(self, seq1, seq2):
        '''Create a matrix of scores representing trial alignments of the two sequences.
    
        Sequence alignment can be treated as a graph search problem. This function
        creates a graph (2D matrix) of scores, which are based on trial alignments
        of different base pairs. The path with the highest cumulative score is the
        best alignment.
        '''        
        # The scoring matrix contains an extra row and column for the gap (-), hence
        # the +1 here.
        n_rows = len(seq1) + 1
        n_cols = len(seq2) + 1
        
        # Initialize with zeros
        self.score_matrix = np.zeros((n_rows,n_cols)) 
    
        # Fill the scoring matrix.
        self.max_score = []
        self.visited_pos = []
        for i in range(1, n_rows):
            for j in range(1, n_cols):
                # Calculate score for a given i, j position in the scoring matrix.
                
                # compare two symbols at (i-1, j-1).
                # do comparison for the case a symbol is a list of symbols (is match if one element of list matches any element of other list)
                symbol1 = seq1[i - 1]
                symbol2 = seq2[j - 1]
                match = False
                if not isinstance(symbol1, list):
                    symbol1 = [symbol1] 
                if not isinstance(symbol2, list):
                    symbol2 = [symbol2]
                for c1 in symbol1:
                    if c1 in symbol2:
                        match = True
                        break
                
                # The score is based on the up, left, and up-left neighbors: always take the max, never go below zero
                up_left_score = self.score_matrix[i - 1][j - 1] + (self.match_score if match else self.mismatch_score)
                up_score      = self.score_matrix[i - 1][j] + self.gap_score
                left_score    = self.score_matrix[i][j - 1] + self.gap_score
                score = max(0, up_left_score, up_score, left_score)
                
                # remember where in the matrix the maximum value was
                if score > 0:
                    heapq.heappush(self.max_score, (-score, (i,j))) # note: we save negative score for descending order of priority queue
    
                self.score_matrix[i][j] = score
    
    
    
    
    def traceback(self, start_pos, seq1, seq2):
        '''Find the optimal path through the matrix.
    
        This function traces a path from the bottom-right to the top-left corner of
        the scoring matrix. Each move corresponds to a match, mismatch, or gap in one
        or both of the sequences being aligned. Moves are determined by the score of
        three adjacent squares: the upper square, the left square, and the diagonal
        upper-left square.
    
        WHAT EACH MOVE REPRESENTS
            diagonal: match/mismatch
            up:       gap in sequence 1
            left:     gap in sequence 2
        '''
        
        def next_move(i,j):
            '''
            find which move gives highest score: 0=diagonal, 1=up, 2=left
            return move index and score value
            '''
            scores = [self.score_matrix[i - 1][j - 1],
                      self.score_matrix[i - 1][j],
                      self.score_matrix[i][j - 1]]
            move = np.argmax(scores)
            return move, scores[move]
    
        DIAG, UP, LEFT = range(3)
        aligned_seq1 = []
        aligned_seq2 = []
        alignment = []
        i, j = start_pos
        move, score = next_move(i,j)
        while score != 0:
            if (i,j) in self.visited_pos:
                return [], '', '', self.visited_pos, False
            self.visited_pos.append((i,j))
            if move == DIAG:
                alignment.append((i-1, j-1))
                aligned_seq1.append(self.string_mapping_function(seq1[i - 1][0] if isinstance(seq1[i - 1], list) else seq1[i - 1]))
                aligned_seq2.append(self.string_mapping_function(seq2[j - 1][0] if isinstance(seq2[j - 1], list) else seq2[j - 1]))
                i -= 1
                j -= 1
            elif move == UP:
                alignment.append((i-1, j))
                aligned_seq1.append(self.string_mapping_function(seq1[i - 1][0] if isinstance(seq1[i - 1], list) else seq1[i - 1]))
                aligned_seq2.append('-')
                i -= 1
            else: # LEFT
                alignment.append((i, j-1))
                aligned_seq1.append('-')
                aligned_seq2.append(self.string_mapping_function(seq2[j - 1][0] if isinstance(seq2[j - 1], list) else seq2[j - 1]))
                j -= 1
    
            move, score = next_move(i, j)
    
        alignment.append((i-1, j-1))
        aligned_seq1.append(self.string_mapping_function(seq1[i - 1][0] if isinstance(seq1[i - 1], list) else seq1[i - 1]))
        aligned_seq2.append(self.string_mapping_function(seq2[j - 1][0] if isinstance(seq2[j - 1], list) else seq2[j - 1]))
        
        alignment.reverse()
        return alignment, ''.join(reversed(aligned_seq1)), ''.join(reversed(aligned_seq2)), True # last True output indicates this alignment is valid, i.e. this position has not been visited yet






