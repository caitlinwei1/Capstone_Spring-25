# Load dataset
dataset_path = "/content/AZ.parquet"
df = pd.read_parquet(dataset_path)

# Ensure NLTK resources are available
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define election-related terms
election_terms = [
    "abortion",
    "international affairs",
    "immigration",
    "economy",
    "violent crime",
    "climate change"
]

# Load a sentence similarity model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Function to get related words using WordNet
def get_related_words(word):
    related_words = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            related_words.add(lemma.name().replace('_', ' '))
    return list(related_words)

# Function to expand related words using semantic similarity
def expand_terms(base_term, top_n=10):
    base_related = get_related_words(base_term)
    all_terms = list(set(base_related + [base_term]))

    if not all_terms:
        return [base_term]  # If no related words are found, return the term itself.

    embeddings = model.encode(all_terms, convert_to_tensor=True)
    topic_embedding = model.encode(base_term, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(topic_embedding, embeddings)[0]
    sorted_indices = similarities.argsort(descending=True)

    expanded_terms = [all_terms[idx] for idx in sorted_indices[:top_n]]
    return list(set(expanded_terms))

# Dictionary to store results
term_analysis = defaultdict(dict)

# Loop over each election term
for term in election_terms:
    related_words = expand_terms(term, top_n=15)  # Expand related words
    term_analysis[term]['related_words'] = related_words  # Store related words

    # Filter dataset for presence of related words
    term_mentions = df[df['text'].str.contains('|'.join(related_words), case=False, na=False)]

    term_analysis[term]['mention_count'] = len(term_mentions)
    term_analysis[term]['examples'] = term_mentions['text'].head(5).tolist()  # Show some examples

# Convert results into DataFrame
# Convert results into DataFrame
results_df = pd.DataFrame.from_dict(term_analysis, orient='index')
results_df.reset_index(inplace=True)
results_df.columns = ['Topic', 'Related Words', 'Mention Count', 'Example Mentions']

# Display results in Google Colab
display(results_df)


