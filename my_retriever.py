from collections import defaultdict
import math
import time    #DEBUG ONLY

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self, index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        
        
        # Number of Documents in Collection - Will be determined as a running total
        self.collection_size = 0
        
        # Dictionary of IDF values (key = term, value = idfValue)
        self.collection_idf_values = defaultdict(float)
        
        # Recording number of elements per document (key = docid, value = number of terms in document)
        self.docid_num_elements = defaultdict(int)
        
        # Recording list of idf + tf values per docid (key = docid, value = list of values)
        self.docid_vector_elements = defaultdict(list)
        
        # Recording Document Vector Magnitude for each docid ([docid] = magnitude)
        self.docid_magnitude = defaultdict(float)
        
        
# ----------------------------------------------------------------------------        
        # Obtaining Collection Size
        set_of_docids = set()
        # Creating docid_num_elements
        
        #DEBUG
        startTime = time.clock()
        
        for term in self.index:
            for docid in self.index[term]:
                set_of_docids.add(docid)
        self.collection_size = len(set_of_docids)
        
        #DEBUG
        print("it took",str(time.clock()-startTime),"to loop through the entire index object and determine how many documents are in the collection")
        
# ----------------------------------------------------------------------------                    
        # Loop through every term in the index and calculating tf & idf per doc and term.
        
        #DEBUG
        startTime = time.clock()
        
        if self.termWeighting == 'tf':
            # find the TF values for every document's terms, and the total number of elements in each document
            # by looping over all of the terms in the entire index
            for term in self.index:
                for docid in self.index[term]:
                    self.docid_num_elements[docid] += self.index[term][docid]
                    self.docid_vector_elements[docid].append(self.index[term][docid])
            
            # calculate the magnitude of the document vectors based on normalised TF elements.
            for (docid, tf_values) in self.docid_vector_elements.items():
                total = 0
                numElems = self.docid_num_elements[docid]
                for elem in tf_values:
                    elem = elem/numElems
                    total += elem * elem
                self.docid_magnitude[docid] = math.sqrt(total)
        
        elif self.termWeighting == 'tfidf':
            # find the TF and IDF values for every document's terms, and the total number of elements in each document
            # by looping over all of the terms in the entire index
            for term in self.index:
                # Calculate the IDF value for this term
                # Calculate dfw - the number of documents the term appears in
                num_occurrences = len(self.index[term])
                idf = math.log10(self.collection_size/num_occurrences)
                self.collection_idf_values[term] = idf
                
                for docid in self.index[term]:
                    self.docid_num_elements[docid] += self.index[term][docid]
                    self.docid_vector_elements[docid].append(self.index[term][docid] * idf)
            
            # calculate the magnitude of the document vectors based on normalised TF elements * idf value.
            for (docid, vector_elems) in self.docid_vector_elements.items():
                total = 0
                numElems = self.docid_num_elements[docid]
                for elem in vector_elems:
                    elem = elem/numElems
                    total += elem * elem
                self.docid_magnitude[docid] = math.sqrt(total)
        
        else:
            # LOGIC FOR BINARY WEIGHTING
            # find the Binary values for every document's terms, and the total number of elements in each document
            # by looping over all of the terms in the entire index
            x = 0
            for term in self.index:
                for docid in self.index[term]:
                    self.docid_num_elements[docid] += 1
                
                #print("*** number of docid elements =", self.docid_num_elements[docid], "for", docid)
           
            
                self.docid_magnitude[docid] = math.sqrt(self.docid_num_elements[docid])
                print("run:", x, ", docid:", docid, "has a docid_magnitude of", self.docid_magnitude[docid] )
                x +=1
            
        
        
        #DEBUG
        print("it took",str(time.clock()-startTime),"to loop through the entire index object and calculate TFs and IDFs")
        
          
# -----------------------------------------------------------------------------                    
    # Method performing retrieval for specified query
    def forQuery(self, query):
        
        # keys are docids, indicating what words are inside it
        # (key = docid, value = cosine similarity score for this query)
        cosine_sim_values = defaultdict(float)
        
        len_query = len(query)
        
        # first run candidate values. This is used to only check documents which have at least one
        # similar term with the query (as all others will have a similarity of zero!)
        # NOTE: WE SHOULD JUST SKIP THINGS IN REAL TIME INSTEAD SINCE WE ARE FORCED TO LOOP AN ENTIRE PASS ANYWAY!
        #candidates = defaultdict(int)
        
        #DEBUG
        #startTime = time.clock()
        
        #DEBUG
        #print("it took",str(time.clock()-startTime),"to build the candidate document set for query:",query)
            
        # looping through each document, and skipping if there are no query terms in doc, 
        # else calculate cosine similarity and add to candidates list [docid] = cosine_sim
        for docid in range(1, self.collection_size + 1):
            
            # numerator of cosine similarity equation
            qd_dot_product = 0
            
            query_i = 0
            doc_i = 0
            
            # sum the squares of each element as we go to eventually figure out the vector magnitude, which will be the denominator
            if self.docid_magnitude[docid] == 0:
                continue
            d_vector_magnitude = self.docid_magnitude[docid]
            #print("d_vector magnitude for", docid, "is", self.docid_magnitude[docid])
            
            for term in query:
                if term in self.index.keys() and docid in self.index[term].keys():
                    if self.termWeighting == 'tf':
                        norm_scale = 1 / self.docid_num_elements[docid]
                        
                        query_i = query[term]/len_query
                        doc_i = self.index[term][docid] * norm_scale
                        
                        
                    elif self.termWeighting == 'tfidf':
                        norm_scale = 1 / self.docid_num_elements[docid]
                        
                        query_i = query[term]/len_query * self.collection_idf_values[term]
                        doc_i = self.index[term][docid] * norm_scale * self.collection_idf_values[term]
                        

                    else:
                        query_i += 1
                        doc_i += 1
                        
                    qd_dot_product += query_i * doc_i
            
            # COSINE SIMILARITY
            cosine_sim = qd_dot_product / d_vector_magnitude
            cosine_sim_values[docid] = cosine_sim
        
        # Finished processing all the documents and obtaining scores
        top_10 = sorted(cosine_sim_values, key = cosine_sim_values.get, reverse = True)[:10]
        print("***top-10:", top_10)
        return top_10
