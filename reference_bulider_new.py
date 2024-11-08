import re
import os
import time
import yake
import nltk
import spacy
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, util
import requests
from transformers import pipeline
import customtkinter as ctk
import bibtexparser
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model for Named Entity Recognition
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer model for sentence similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class ReferenceHelper:
    def __init__(self, summarization_model="facebook/bart-large-cnn", yake_n=2, yake_dedup_lim=0.9, yake_top=5):
        self.summarization_pipeline = pipeline(
            "summarization", 
            model=summarization_model, 
            device=0 if torch.cuda.is_available() else -1
        )
        # Set up YAKE keyword extractor with customizable parameters
        self.kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=yake_n, 
            dedupLim=yake_dedup_lim, 
            top=yake_top
        )

    def generate_search_query(self, sentence):
        # Summarize sentence to get the core idea
        summary = self.summarization_pipeline(sentence, max_length=30, min_length=10, do_sample=False)
        summarized_text = summary[0]['summary_text']

        # Extract keywords using YAKE
        keywords = self.kw_extractor.extract_keywords(summarized_text)
        
        # Formulate the search query from the keywords, adding specific terms for academic research
        query = ' '.join([kw[0] for kw in keywords]) + " academic research paper"
        
        # Limit the query to 100 characters for compatibility with academic search tools
        return query[:100]

    def generate_citation_key(self, bib_entry):
        # Extract citation key from BibTeX entry
        match = re.search(r'@\w+{(.*?),', bib_entry)
        if match:
            return match.group(1)
        return None

    def generate_unique_citation_key(self, title, author):
        # Generate a unique citation key based on title and author
        title_key = ''.join(word[:3] for word in title.split()[:2]).lower()
        author_key = author.split()[0].lower() if author else 'unknown'
        return f"{author_key}_{title_key}"

    def search_academic_sources(self, query, top_k=20, retries=3):
        # Use CrossRef API to search for DOIs
        results = []
        for attempt in range(retries):
            try:
                response = requests.get(
                    "https://api.crossref.org/works",
                    params={"query": query, "rows": top_k},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if "message" in data and "items" in data["message"] and len(data["message"]["items"]) > 0:
                        for item in data["message"]["items"][:top_k]:
                            doi = item.get("DOI")
                            title = item.get("title", ["No Title"])[0]
                            authors = ', '.join([author.get("family", "") for author in item.get("author", [])])
                            abstract = item.get("abstract", "")
                            if doi:
                                results.append({"doi": doi, "title": title, "authors": authors, "abstract": abstract})
                    return results
                return results
            except requests.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Error contacting CrossRef API: {e}")
                    return results
        return results

    def convert_doi_to_bib(self, doi):
        # Convert DOI to BibTeX using CrossRef API
        try:
            headers = {"Accept": "application/x-bibtex"}
            response = requests.get(f"https://doi.org/{doi}", headers=headers)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Error converting DOI to BibTeX: {e}")
        return None

class DocumentProcessor:
    def __init__(self, tex_path, bib_path, similarity_threshold=0.8, yake_n=2, yake_dedup_lim=0.9, yake_top=5, search_top_k=20, display_top_k=3):
        self.tex_path = tex_path
        self.bib_path = bib_path
        self.helper = ReferenceHelper(yake_n=yake_n, yake_dedup_lim=yake_dedup_lim, yake_top=yake_top)
        self.bib_entries = self.load_bib_entries()
        self.similarity_threshold = similarity_threshold
        self.search_top_k = search_top_k
        self.display_top_k = display_top_k

    def load_bib_entries(self):
        # Load BibTeX entries from the .bib file
        bib_entries = {}
        if os.path.exists(self.bib_path):
            with open(self.bib_path, 'r', encoding='utf-8') as bib_file:
                content = bib_file.read()
                entries = re.findall(r'@\w+{.*?,.*?\n}', content, re.DOTALL)
                for entry in entries:
                    key = self.helper.generate_citation_key(entry)
                    if key:
                        bib_entries[key] = entry
        return bib_entries

    def extract_sentences_from_tex(self):
        # Read the content of the .tex file
        with open(self.tex_path, 'r', encoding='utf-8') as tex_file:
            tex_content = tex_file.read()

        # Remove LaTeX commands to focus on the text
        clean_text = re.sub(r'\\[a-zA-Z]+{.*?}|%.*?\n', '', tex_content)  # Removes LaTeX commands and comments

        # Split text into sentences
        sentences = sent_tokenize(clean_text)
        return sentences

    def process_sentences(self):
        sentences = self.extract_sentences_from_tex()
        sentences_needing_citations = []
        existing_sentences = []

        # Extended citation patterns and statistical indicators
        citation_phrases = [
            r'\b(has been shown|studies|evidence|research|suggests|observations|data shows|statistics indicate|found)\b',
            r'\b(as demonstrated in|as seen in|described by|reported by)\b',
            r'\b(recent studies|recent findings|significant results|in this study|our approach)\b',
            r'\b(\d+%|significant|correlation|p-value|statistical analysis|clinical trials)\b'
        ]
        citation_pattern = re.compile('|'.join(citation_phrases), re.IGNORECASE)

        for sentence in sentences:
            doc = nlp(sentence)
            contains_entities = any(ent.label_ in ["ORG", "PERSON", "GPE", "DATE", "EVENT", "WORK_OF_ART"] for ent in doc.ents)
            
            # Determine if a sentence needs citation
            needs_citation = bool(citation_pattern.search(sentence))
            contains_numbers = bool(re.search(r'\b\d+(\.\d+)?\b', sentence))
            starts_with_demonstrative = sentence.strip().lower().startswith(('this', 'these', 'such'))
            
            # Avoid redundancy with similarity checks
            is_similar = False
            if existing_sentences:
                embeddings = embedding_model.encode([sentence] + existing_sentences)
                similarity_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])
                is_similar = max(similarity_scores[0]) > self.similarity_threshold
            
            if (contains_entities or needs_citation or contains_numbers or starts_with_demonstrative) and not is_similar:
                sentences_needing_citations.append(sentence)
                existing_sentences.append(sentence)
                file_path = os.path.join(os.path.dirname(self.tex_path), "sentences.txt")
                
        return sentences_needing_citations


    def update_bib_file(self, bib_entry):
        # Update or create BibTeX entries based on the bib_entry
        with open(self.bib_path, 'a', encoding='utf-8') as bib_file:
            # Extract title and author for generating a unique citation key
            match = re.search(r'title\s*=\s*{(.+?)}', bib_entry, re.IGNORECASE)
            title = match.group(1) if match else "unknown"
            match = re.search(r'author\s*=\s*{(.+?)}', bib_entry, re.IGNORECASE)
            author = match.group(1) if match else "unknown"

            citation_key = self.helper.generate_unique_citation_key(title, author)
            if citation_key not in self.bib_entries:
                # Add the citation key to the BibTeX entry
                updated_entry = re.sub(r'@\w+{', f'@article{{{citation_key}, ', bib_entry, count=1)
                self.bib_entries[citation_key] = updated_entry
                bib_file.write(f"\n{updated_entry}")
                print(f"Added BibTeX entry for citation key: {citation_key}")
            else:
                print(f"Citation key {citation_key} already exists in BibTeX file.")

    def insert_citation_into_tex(self, sentence, citation_key):
        # Read the .tex file
        with open(self.tex_path, 'r', encoding='utf-8') as tex_file:
            tex_content = tex_file.read()

        # Find the sentence in the tex_content
        # Note: This is a simplistic approach; for more accurate insertion, consider parsing the LaTeX structure
        pattern = re.escape(sentence)
        match = re.search(pattern, tex_content)
        if match:
            # Insert the citation after the sentence
            insertion_point = match.end()
            tex_content = tex_content[:insertion_point] + f" \\cite{{{citation_key}}}" + tex_content[insertion_point:]
            
            # Write back the updated .tex file
            with open(self.tex_path, 'w', encoding='utf-8') as tex_file:
                tex_file.write(tex_content)
            print(f"Inserted citation \\cite{{{citation_key}}} into .tex file.")
        else:
            print("Sentence not found in .tex file; citation not inserted.")

    def calculate_similarity(self, sentence, reference):
        # Combine sentence and reference text
        sentence_clean = clean_text(sentence)
        reference_clean = clean_text(reference['title'] + ' ' + reference.get('abstract', ''))

        # Vectorize using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([sentence_clean, reference_clean])

        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity

    def process_document(self, gui_callback=None):
        # Full processing routine
        print("Starting document processing...")
        sentences_needing_citations = self.process_sentences()
        print(f"Found {len(sentences_needing_citations)} sentences needing citations.")
        
        # For each sentence, find top_k references and calculate similarity scores
        results = []
        for sentence in sentences_needing_citations:
            query = self.helper.generate_search_query(sentence)
            top_refs = self.helper.search_academic_sources(query, top_k=self.search_top_k)
            scored_refs = []
            for ref in top_refs:
                similarity = self.calculate_similarity(sentence, ref)
                ref['similarity'] = similarity
                scored_refs.append(ref)
            # Sort references by similarity score in descending order
            scored_refs = sorted(scored_refs, key=lambda x: x['similarity'], reverse=True)
            # Keep only top 3 references
            top_display_refs = scored_refs[:self.display_top_k]
            results.append({"sentence": sentence, "references": top_display_refs})
        
        if gui_callback:
            gui_callback(results)
        
        print("Document processing completed.")

class ReferenceAnalyzerGUI:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("Reference Analyzer")
        self.root.geometry("1200x800")
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="Reference Analyzer", 
            font=("Helvetica", 24, "bold")
        )
        self.title_label.pack(pady=10)
        
        # File selection frame
        self.file_frame = ctk.CTkFrame(self.main_frame)
        self.file_frame.pack(fill="x", padx=10, pady=10)
        
        # LaTeX file selection
        self.tex_label = ctk.CTkLabel(self.file_frame, text="LaTeX File:")
        self.tex_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.tex_entry = ctk.CTkEntry(self.file_frame, width=600)
        self.tex_entry.grid(row=0, column=1, padx=5, pady=5)
        self.tex_button = ctk.CTkButton(self.file_frame, text="Browse", command=self.browse_tex)
        self.tex_button.grid(row=0, column=2, padx=5, pady=5)
        
        # BibTeX file selection
        self.bib_label = ctk.CTkLabel(self.file_frame, text="BibTeX File:")
        self.bib_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.bib_entry = ctk.CTkEntry(self.file_frame, width=600)
        self.bib_entry.grid(row=1, column=1, padx=5, pady=5)
        self.bib_button = ctk.CTkButton(self.file_frame, text="Browse", command=self.browse_bib)
        self.bib_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Customization options
        self.options_frame = ctk.CTkFrame(self.main_frame)
        self.options_frame.pack(fill="x", padx=10, pady=10)
        
        # Similarity Threshold
        self.threshold_label = ctk.CTkLabel(self.options_frame, text="Similarity Threshold:")
        self.threshold_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.threshold_slider = ctk.CTkSlider(
            self.options_frame, 
            from_=0, 
            to=1, 
            number_of_steps=100, 
            command=self.update_threshold_label
        )
        self.threshold_slider.set(0.8)
        self.threshold_slider.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        self.threshold_value_label = ctk.CTkLabel(self.options_frame, text="0.80")
        self.threshold_value_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        # YAKE Parameters
        # YAKE N-gram Size
        self.yake_n_label = ctk.CTkLabel(self.options_frame, text="YAKE N-gram Size:")
        self.yake_n_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.yake_n_entry = ctk.CTkEntry(self.options_frame, width=50)
        self.yake_n_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.yake_n_entry.insert(0, "2")
        
        # YAKE Deduplication Limit
        self.yake_dedup_lim_label = ctk.CTkLabel(self.options_frame, text="YAKE Deduplication Limit:")
        self.yake_dedup_lim_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.yake_dedup_lim_entry = ctk.CTkEntry(self.options_frame, width=50)
        self.yake_dedup_lim_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.yake_dedup_lim_entry.insert(0, "0.9")
        
        # YAKE Top Keywords Count
        self.yake_top_label = ctk.CTkLabel(self.options_frame, text="YAKE Top Keywords Count:")
        self.yake_top_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.yake_top_entry = ctk.CTkEntry(self.options_frame, width=50)
        self.yake_top_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.yake_top_entry.insert(0, "5")
        
        # Number of References to Fetch
        self.search_top_k_label = ctk.CTkLabel(self.options_frame, text="Number of References to Fetch:")
        self.search_top_k_label.grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.search_top_k_entry = ctk.CTkEntry(self.options_frame, width=50)
        self.search_top_k_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.search_top_k_entry.insert(0, "20")
        
        # Configure grid weights
        self.options_frame.columnconfigure(1, weight=1)
        
        # Analyze button
        self.analyze_button = ctk.CTkButton(self.main_frame, text="Analyze References", command=self.analyze)
        self.analyze_button.pack(pady=10)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(self.main_frame)
        self.progress.pack(fill="x", padx=10, pady=10)
        self.progress.set(0)
        
        # Results display
        self.results_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Selected references storage
        self.selected_references = {}  # Key: sentence, Value: selected reference
        
        self.root.mainloop()
    
    def update_threshold_label(self, value):
        self.threshold_value_label.configure(text=f"{float(value):.2f}")
    
    def browse_tex(self):
        filename = filedialog.askopenfilename(filetypes=[("LaTeX files", "*.tex")])
        if filename:
            self.tex_entry.delete(0, "end")
            self.tex_entry.insert(0, filename)
    
    def browse_bib(self):
        filename = filedialog.askopenfilename(filetypes=[("BibTeX files", "*.bib")])
        if filename:
            self.bib_entry.delete(0, "end")
            self.bib_entry.insert(0, filename)
    
    def analyze(self):
        tex_file = self.tex_entry.get()
        bib_file = self.bib_entry.get()
        threshold = self.threshold_slider.get()
        
        if not os.path.isfile(tex_file) or not os.path.isfile(bib_file):
            messagebox.showerror("Error", "Please select valid LaTeX and BibTeX files.")
            return
        
        try:
            yake_n = int(self.yake_n_entry.get())
            yake_dedup_lim = float(self.yake_dedup_lim_entry.get())
            yake_top = int(self.yake_top_entry.get())
            search_top_k = int(self.search_top_k_entry.get())
            if search_top_k < 3:
                raise ValueError("Number of references to fetch should be at least 3.")
        except ValueError as ve:
            messagebox.showerror("Error", f"Please enter valid parameters.\n{ve}")
            return
        
        # Initialize DocumentProcessor
        self.processor = DocumentProcessor(
            tex_path=tex_file,
            bib_path=bib_file,
            similarity_threshold=threshold,
            yake_n=yake_n,
            yake_dedup_lim=yake_dedup_lim,
            yake_top=yake_top,
            search_top_k=search_top_k,
            display_top_k=3  # Fixed to 3 as per requirements
        )
        
        # Disable the analyze button to prevent multiple clicks
        self.analyze_button.configure(state="disabled")
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.results_frame.update_idletasks()
        
        self.progress.set(0.1)
        self.results_frame.update_idletasks()
        
        # Start processing in a separate thread to keep the GUI responsive
        thread = threading.Thread(target=self.processor.process_document, args=(self.display_results,))
        thread.start()
    
    def display_results(self, results):
        self.progress.set(0.5)
        self.results_frame.update_idletasks()
        
        if not results:
            no_sentences_label = ctk.CTkLabel(
                self.results_frame, 
                text="No sentences requiring citations were found.",
                wraplength=1100,
                justify="left",
                fg_color="transparent"
            )
            no_sentences_label.pack(anchor="w", pady=10)
            self.progress.set(1.0)
            self.analyze_button.configure(state="normal")
            return
        
        for idx, item in enumerate(results, start=1):
            sentence = item["sentence"]
            references = item["references"]
            
            # Display the sentence
            sentence_label = ctk.CTkLabel(
                self.results_frame, 
                text=f"Sentence {idx}:\n{sentence}", 
                font=("Helvetica", 12, "bold"),
                wraplength=1100,
                justify="left"
            )
            sentence_label.pack(anchor="w", pady=(10, 5))
            
            if references:
                # Create a frame for references
                ref_frame = ctk.CTkFrame(self.results_frame)
                ref_frame.pack(fill="x", padx=20, pady=5)
                
                for ref_idx, ref in enumerate(references, start=1):
                    ref_container = ctk.CTkFrame(ref_frame)
                    ref_container.pack(fill="x", pady=2)
                    
                    # Reference details with similarity score
                    ref_details = (
                        f"{ref_idx}. {ref['title']} by {ref['authors']} (DOI: {ref['doi']})\n"
                        f"Similarity Score: {ref['similarity']:.4f}"
                    )
                    ref_label = ctk.CTkLabel(
                        ref_container, 
                        text=ref_details, 
                        wraplength=850, 
                        justify="left"
                    )
                    ref_label.pack(side="left", padx=5, pady=5)
                    
                    # Selection button
                    select_button = ctk.CTkButton(
                        ref_container, 
                        text="Select",
                        command=lambda s=sentence, r=ref: self.select_reference(s, r)
                    )
                    select_button.pack(side="right", padx=5, pady=5)
            else:
                no_ref_label = ctk.CTkLabel(
                    self.results_frame, 
                    text="No references found for this sentence.",
                    fg_color="transparent",
                    text_color="red",
                    wraplength=1100,
                    justify="left"
                )
                no_ref_label.pack(anchor="w", padx=20, pady=5)
        
        self.progress.set(1.0)
        self.analyze_button.configure(state="normal")
        messagebox.showinfo("Analysis Complete", "Reference analysis completed. Please review and select appropriate references.")
    
    def select_reference(self, sentence, reference):
        # Store the selected reference
        self.selected_references[sentence] = reference
        
        # Disable all select buttons for this sentence
        for widget in self.results_frame.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and widget.cget("text").startswith("Sentence"):
                if sentence in widget.cget("text"):
                    # Find the next sibling frame which contains the references
                    ref_frame = widget.master.master
                    for child in ref_frame.winfo_children():
                        if isinstance(child, ctk.CTkFrame):
                            for ref_container in child.winfo_children():
                                for sub_child in ref_container.winfo_children():
                                    if isinstance(sub_child, ctk.CTkButton) and sub_child.cget("text") == "Select":
                                        sub_child.configure(state="disabled")
        
        # Update the GUI to reflect selection
        selection_label = ctk.CTkLabel(
            self.results_frame, 
            text=(
                f"Selected Reference for the above sentence:\n"
                f"{reference['title']} by {reference['authors']} (DOI: {reference['doi']})\n"
            ),
            fg_color="transparent",
            text_color="green",
            wraplength=1100,
            justify="left"
        )
        selection_label.pack(anchor="w", padx=20, pady=(0,10))
        
        # Proceed to update BibTeX and .tex files
        threading.Thread(target=self.update_references, args=(sentence, reference)).start()
    
    def update_references(self, sentence, reference):
        # Convert DOI to BibTeX
        bib_entry = self.processor.helper.convert_doi_to_bib(reference['doi'])
        if bib_entry:
            # Update BibTeX file
            self.processor.update_bib_file(bib_entry)
            
            # Generate citation key
            match = re.search(r'title\s*=\s*{(.+?)}', bib_entry, re.IGNORECASE)
            title = match.group(1) if match else "unknown"
            match = re.search(r'author\s*=\s*{(.+?)}', bib_entry, re.IGNORECASE)
            author = match.group(1) if match else "unknown"
            citation_key = self.processor.helper.generate_unique_citation_key(title, author)
            
            # Insert citation into .tex file
            self.processor.insert_citation_into_tex(sentence, citation_key)
            
            # Notify the user
            messagebox.showinfo(
                "Reference Added", 
                f"Reference '{title}' has been added to the BibTeX file and cited in the LaTeX document."
            )
        else:
            messagebox.showerror(
                "Error", 
                f"Failed to convert DOI {reference['doi']} to BibTeX."
            )

# Helper functions
def extract_text_from_tex(tex_file):
    with open(tex_file, 'r', encoding='utf-8', errors='ignore') as file:
        tex_content = file.read()
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', tex_content)
    text = re.sub(r'\\[a-zA-Z]+\s?', '', text)
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'%.*', '', text)
    text = re.sub(r'\{|\}', '', text)
    return text

def parse_bib_file(bib_file):
    with open(bib_file, 'r', encoding='utf-8', errors='ignore') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    references = {}
    for entry in bib_database.entries:
        key = entry.get('ID', 'NoID')
        title = entry.get('title', '')
        abstract = entry.get('abstract', '')
        keywords = entry.get('keywords', '')
        combined_text = ' '.join([title, abstract, keywords])
        references[key] = combined_text
    return references

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered_words)

def calculate_similarity(doc_text, references):
    doc_text_clean = clean_text(doc_text)
    ref_texts_clean = {key: clean_text(value) for key, value in references.items()}
    all_texts = [doc_text_clean] + list(ref_texts_clean.values())
    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    vectors = vectorizer.toarray()
    doc_vector = vectors[0]
    ref_vectors = vectors[1:]
    similarities = cosine_similarity([doc_vector], ref_vectors)[0]
    return dict(zip(ref_texts_clean.keys(), similarities))

def evaluate_relevance(similarity_scores, threshold=0.1):
    relevant_refs = {key: score for key, score in similarity_scores.items() if score >= threshold}
    non_relevant_refs = {key: score for key, score in similarity_scores.items() if score < threshold}
    return relevant_refs, non_relevant_refs

if __name__ == "__main__":
    ReferenceAnalyzerGUI()
