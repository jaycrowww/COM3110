from collections import defaultdict
from math import *

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self, index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        
        
        # initialise precomputed processes
        self.idf_values = defaultdict(int)
        
        # Dictionary of documents and the number of elements they contain
    
        self.docid_num_elements = defaultdict(int)
        self.document_vector_sizes = defaultdict(int)
        self.collection_size = 0
        
        
        # Creating Document Vector Size Dict
        
        # Creating docid_num_elements
        for term in self.index:
            # gets docid
            for docid in self.index[term]:
                self.docid_num_elements[docid] += 1
                                
                """if self.termWeighting == 'binary':
                    self.document_vector_sizes[docid] += 1
                
                # Document Sizes with Term Frequency Consideration
                else:
                    # print("term[entry]:", term[entry])
                    self.document_vector_sizes[docid] += self.index[term][docid]/len()"""
                    
                if self.termWeighting == 'binary':
                    self.document_vector_sizes[docid] += 1
        print("++UPDATE: document vector size for", 4, "is", self.document_vector_sizes[4])

            
        
        # Calculate Collection Size
        for docid in self.docid_num_elements:
            self.collection_size += 1
        print("*** Collection size = " + str(self.collection_size)) 
        
        # Calculate IDF        
        if self.termWeighting == 'tfidf':
            for term in self.index:
                # Calculate dfw
                num_occurrences = len(self.index[term])
                #print("** num_ocurrences:", num_occurrences)
                
                idf = log10(self.collection_size/num_occurrences)
                
                self.idf_values[term] = idf
                
        #print("idf values:", self.idf_values)
        
        # Creating document_vector_sizes for tf & tfidf
        if self.termWeighting != 'binary':
            
            # Iterates through index to obtain document vector sizes relative to term weighting
            for term in self.index:

                for docid in self.index[term]:
                    # Obtains Term Frequency Value 
                    doc_tf = self.index[term][docid]/self.docid_num_elements[docid]
                    
                    if self.termWeighting == 'tfidf':
                        doc_tfidf = doc_tf * self.idf_values[term]
                        self.document_vector_sizes[docid] += doc_tfidf
                        
                    else:
                        self.document_vector_sizes[docid] += doc_tf
                   
                    
                
        
        
        
                

    # Method performing retrieval for specified query
    def forQuery(self, query):
        
        # keys are docids, indicating what words are inside it
        similarity_values = defaultdict(list)
        
        # first run candidate values
        first_run_candidates = set()
        
        # map docids to overall similarity values
        candidate_values = defaultdict(float)
        
        #
        qd_val = 0
        
        # Obtaining candidates that contain at least one query term
        for term in query:
            # tests validity
            if term in self.index.keys():
                for docid in self.index[term]:
                    first_run_candidates.add(docid)
        
        #number_of_candidates = len(first_run_candidates)
            
        # iterating first run candidates and calculating similarity scores
        
        for docid in first_run_candidates:
            qd_val = 0
            #print("+++ docid test:", i)
            for term in query:
                if term in self.index:
                    for check_doc in self.index[term]:
                        if check_doc == docid:
                            #print("### print check_doc val:", check_doc)
                            if self.termWeighting == 'binary':
                                # CODE value of qd for binary
                                qd_val += 1
                                
                            else: 
                                # CODE value of qd for tf
                                doc_i = self.index[term][docid]/self.docid_num_elements[docid]
                                query_i = query[term]/len(query)
                                
                                tf_val = doc_i * query_i
                                
    
                                if self.termWeighting == "tfidf":
                                    # CODE value of qd for tfidf
                                    qd_val += tf_val * self.idf_values[term]
                                    
                                # if term weighting = "tf"    
                                else:
                                    qd_val += tf_val
                                
                    # COSINE SIMILARITY
            cosine_sim = qd_val / self.document_vector_sizes[docid]
           # print("docid:", docid)
           # print("**cosine sim:", cosine_sim)
                            
                
            
                
            
            #print("set of first run candidates:", first_run_candidates)            
            #print("length of first run candidates:", len(first_run_candidates))
            
                
        
                  
        
            
        
        
        
        # if self.termWeighting == function name, then run function
        
            # binary - whether or not term is present - consideration of
            # multiple occurences 
            
            # tf (term frequency) - frequency of term in document - considerati
            # on of how frequent in collection
            
            # tfidf
            
                # obtain frequent words in document (number of term uses/total number of words in doc)
                
                # obtain inverse - log (size of collection/number of docs containing term)
                
                # tf.idf 
        
        
        return range(1,11)
    
