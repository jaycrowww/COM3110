from collections import defaultdict
import math

class Retrieve:
    
    """" 
    Takes an inverted index of a document collection, a term weighting scheme 
    ('binary', 'tf' or 'tfidf') and a query, and returns the ids of ten top 
    documents based on their cosine similarity to the query.
    """
    
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self, index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        
        
        # Total Number of Documents in Collection - will be determined as a running total
        self.collection_size = 0
        
        # Dictionary of IDF values of Collection (key = term, value = idfValue)
        self.collection_idf_values = defaultdict(float)
        
        # Recording number of elements per document (key = docid, value = total number of terms in document)
        self.docid_num_elements = defaultdict(int)
        
        # Recording list of vector values per docid (key = docid, value = list of values)
        self.docid_vector_elements = defaultdict(list)
        
        # Recording Document Vector Magnitude for each docid ([docid] = magnitude)
        self.docid_magnitude = defaultdict(float)
        

# ----------------------------------------------------------------------------        
        # Calculating Collection Size by storing unique docids
        set_of_docids = set()
            
        for term in self.index:
            for docid in self.index[term]:
                set_of_docids.add(docid)
        self.collection_size = len(set_of_docids)
        
# ----------------------------------------------------------------------------                    
        # Loop through every term in the index and calculating Document Vector
        # Magnitude per document for given term weighting
        
        
        # Term Frequency Approach
        if self.termWeighting == 'tf':
            # find the TF values for every document's terms, and the total
            # number of elements in each document by looping over all of the
            # terms in the entire index
            for term in self.index:
                for docid in self.index[term]:
                    self.docid_num_elements[docid] += self.index[term][docid]
                    self.docid_vector_elements[docid].append(self.index[term][docid])
            
            # calculate the magnitude of the document vectors based on normalised TF elements.
            for (docid, tf_values) in self.docid_vector_elements.items():
                total = 0
                num_elems = self.docid_num_elements[docid]
                for elem in tf_values:
                    elem = elem/num_elems
                    total += elem * elem
                self.docid_magnitude[docid] = math.sqrt(total)
        
        # TFIDF Approach
        elif self.termWeighting == 'tfidf':
            # find the TF and IDF values for every document's terms, and the 
            # total number of elements in each document by looping over all of
            # the terms in the entire index
            for term in self.index:
            # Calculate the IDF value for this term
                num_occurrences = len(self.index[term])
                idf = math.log10(self.collection_size/num_occurrences)
                
                # Store idf value of term in collection for later use
                self.collection_idf_values[term] = idf
                
                for docid in self.index[term]:
                    self.docid_num_elements[docid] += self.index[term][docid]
                    self.docid_vector_elements[docid].append(self.index[term][docid] * idf)
            
            # calculate the magnitude of the document vectors based on 
            # normalised TF elements * idf value.
            for (docid, vector_elems) in self.docid_vector_elements.items():
                total = 0
                num_elems = self.docid_num_elements[docid]
                for elem in vector_elems:
                    elem = elem/num_elems
                    total += elem * elem
                self.docid_magnitude[docid] = math.sqrt(total)
        
        # Binary Approach
        else:
            # find the Binary values for every document's terms, and the total 
            # number of elements in each document by looping over all of the
            # terms in the entire index
            for term in self.index:
                for docid in self.index[term]:
                    self.docid_num_elements[docid] += 1
                    
                self.docid_magnitude[docid] = math.sqrt(self.docid_num_elements[docid])
            
# -----------------------------------------------------------------------------                    
    # Method performing retrieval for specified query
    def forQuery(self, query):

        # Recording cosine similarity values for each document
        # (key = docid, value = cosine similarity score for this query)
        cosine_sim_values = defaultdict(float)
        
        # Number of terms in the query (length)
        len_query = len(query)
        
        # looping through each document, and skipping if there are no query terms in doc, 
        # else calculate cosine similarity and add to candidates list [docid] = cosine_sim
        for docid in range(1, self.collection_size + 1):
            
            # Skip documents with document vector magnitude of zero
            if self.docid_magnitude[docid] == 0:
                continue
            
            # numerator of cosine similarity equation
            qd_dot_product = 0
            
            # sub-parts of numerator of cosine similarity equation
            query_i = 0
            doc_i = 0
            
            
            # sum the squares of each element as we go to eventually figure 
            # out the vector magnitude, which will be the denominator
            d_vector_magnitude = self.docid_magnitude[docid]
            
            for term in query:
                if term in self.index.keys() and docid in self.index[term].keys():
                    
                    # Term Frequency Approach
                    if self.termWeighting == 'tf':
                        
                        # Used to normalised TF values from preprocessing stage
                        # by dividing by total number of elements of docid
                        norm_scale = 1 / self.docid_num_elements[docid]
                        
                        query_i = query[term]/len_query
                        doc_i = self.index[term][docid] * norm_scale
                        
                    
                    # TFIDF Approach
                    elif self.termWeighting == 'tfidf':
                        norm_scale = 1 / self.docid_num_elements[docid]
                        
                        query_i = query[term]/len_query * self.collection_idf_values[term]
                        doc_i = self.index[term][docid] * norm_scale * self.collection_idf_values[term]
                        
                    # Binary Approach
                    else:
                        # only matters whether term appears or not (0 or 1)
                        query_i += 1
                        doc_i += 1
                    
                    # Calculation of Numerator
                    qd_dot_product += query_i * doc_i
            
            # Cosine Similarity Calculation
            cosine_sim = qd_dot_product / d_vector_magnitude
            
            # Recording Cosine Similarity Values per docid
            cosine_sim_values[docid] = cosine_sim
        
        # Finished processing all the documents and obtaining scores
        # returns top 10 ranked results
        top_10 = sorted(cosine_sim_values, key = cosine_sim_values.get, reverse = True)[:10]
        return top_10
